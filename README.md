# Hidden Gem Radio — Deep Music Theory DNA

深層音楽理論DNA（T1〜T6）× Contextual Bandit（MABWiser LinUCB）× YouTube視聴履歴「頻度×直近性」重み付けにより、「未聴の隠れた名曲」を TuneIn 風の連続ラジオ形式で流し続けるローカル音楽ディグシステムです（Android Termux + Brave 動作対応・追加ネイティブビルド不要）。

特定プラットフォームの廃止済み推薦API（Spotify `/v1/recommendations`・`/v1/audio-features` 等）には一切依存せず、公開音楽メタデータ（ListenBrainz / MusicBrainz）＋ `ytmusicapi` ＋ OSS Contextual Bandit の最小構成で動作します。

## 5 Stations (Deep Music Theory DNA)

[`config/music_theory_profile.yaml`](config/music_theory_profile.yaml) に 6 軸（T1 不在トニック循環 / T2 トレシーロ跳ね / T3 和声推進力 / T4 明暗コントラスト爆発 / T5 即効パルス＆フック / T6 履歴親和×隠れ度）と 5 局を定義：

1. `us_hiphop_bouncy` — **US Hip-Hop & Bouncy Rap**
2. `japan_hiphop_street` — **Japan Hip-Hop & Street**
3. `edm_dance_party` — **EDM & Dance Party**
4. `melodic_rap_alt` — **Melodic Rap & Alt-Pop**
5. `indie_groove_hooks` — **Indie Groove & Feel-Good**

## Key Features

- **10次元音楽理論DNAエンジン (`src/recommender/theory_features.py` & `bandit.py`):**
  - `[t1_tonal_suspension, t2_tresillo_bounce, t3_harmonic_propulsion, t4_valence_contrast, t5_pulse_hook, yt_history_affinity, hidden_gem_score, cross_genre_bridge_score, tod_sin, tod_cos]` を `MABWiser`（LinUCB）へ投入し、Good (+1.0) / BAD (-1.0) からユーザーの理論DNA選好をオンライン学習。
  - `GET /api/stations/{id}/next`・`/batch` で `theory_explanation`（日本語解説＋上位2軸）を返却し、`GET /api/profile/theory` でリアルタイムの T1〜T5 嗜好バーを表示。
- **YouTube視聴履歴「頻度 × 直近性」インポーター (`src/digger/history_importer.py`):**
  - `weight = log1p(play_count) * exp(-days_ago / 120.0)` で重み付けして 5 局へ自動分類。
  - ヘビロテ曲本人（`known_video_ids` / `known_track_keys`）の再出題を避けつつ、高重みシードの近傍にある Hidden Gems を最優先で発掘。
- **再生詰まり（0:00 / 0:00）7秒ウォッチドッグ ＆ 30日ブラックリスト:**
  - 7秒以内に再生開始しない／エラー（2, 5, 100, 101, 150）が出た動画は **BAD 学習を汚さず** 自動スキップし、`POST /api/tracks/{video_id}/unplayable` で SQLite に 30 日間ブラックリスト登録。
- **Good プレイリスト保存 ＆ YouTube ハイブリッド連携:**
  - 4種の初期プレイリスト＋新規作成に対応し、認証不要の「YouTube でプレイリストを開く」（`watch_videos?video_ids=...` 50曲分割）と、`data/browser.json` / `data/oauth.json` 認証時の直接 `add_playlist_items()`＋`rate_song(video_id, 'LIKE')` 同期に対応。
- **Android（Termux + Brave）最適化:**
  - Media Session API（ロック画面・イヤホン操作）、スマホ縦画面向け下部固定コントロールバー、[`scripts/start_android.sh`](scripts/start_android.sh)（`termux-wake-lock` 自動付与＆終了時アンロック）。

## Quickstart

```bash
pip install -r requirements.txt

# (任意) data/watch-history.json や data/browser.json を配置して 5 局を重み付け更新
python3 -m src.digger.history_importer --stations 5 --offline

# デスクトップ起動
uvicorn src.radio_server:app --host 127.0.0.1 --port 8000 --reload

# Android (Termux + Brave) 起動
./scripts/start_android.sh
```

## Test & Verification

```bash
pytest -v
ruff check src tests
bash -n scripts/start_android.sh
```
