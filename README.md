# Hidden Gem Radio

お気に入りのジャンル・アーティストから「未聴の隠れた名曲」を自動発掘し、TuneIn風の連続ラジオ形式で再生しながら **Good / BAD** 評価を **Contextual Bandit（MABWiser LinUCB）** へ即時フィードバックするローカル音楽ディグシステム（Phase 1 MVP）。

特定プラットフォームの廃止済み推薦API（Spotify `/v1/recommendations`・`/v1/audio-features` 等）には一切依存せず、公開音楽メタデータ（ListenBrainz / MusicBrainz）＋ `ytmusicapi` ＋ OSS Contextual Bandit の最小構成で動作します。

## Architecture

- **Platform-Agnostic Core (`src/models.py`, `src/db.py`):** 楽曲を `ISRC` / `MBID` / `(artist, title)` 正規化キーでSQLite管理。30日以内のBADアーティスト除外・再生数上限フィルタ・`unplayable`フラグを統合。
- **Adapters (`src/adapters/`):**
  - Phase 1: `YouTubeMusicAdapter` (`ytmusicapi` 検索 + Watch Playlist 展開)
  - Phase 2 (Extension Point): `SpotifyAdapter`（OAuth + Web Playback SDK / プレイリスト自動同期）
  - Phase 3 (Extension Point): `AppleMusicAdapter`（Apple MusicKit JS）
- **Stage 1 Digging & History Importer (`src/digger/`):**
  - `history_importer.py`: YouTube / YouTube Music 視聴履歴・高評価から自動で 2〜4 局の `config/stations.yaml` を生成（未配置時は仕様例シードへフォールバック）。
  - `listenbrainz.py` + `pool.py`: ListenBrainz / MusicBrainz タグ・類似アーティスト + `ytmusicapi` Watch Playlist から 50〜100 曲の候補を SQLite にキャッシュ。
- **Stage 2 Contextual Bandit (`src/recommender/bandit.py`):**
  - `MABWiser` (`LearningPolicy.LinUCB`) によるリアルタイム再ランキング＆ローカル永続化（`data/bandit_<station_id>.pkl`）。
  - 特徴量: `[tag_one_hot, release_decade, log_view_bucket, time_of_day_bucket, is_new_artist]`
  - 報酬: `GOOD = 1.0`, `COMPLETE (>=80%) = 0.7`, `MID_SKIP = 0.3`, `BAD / EARLY_SKIP (<15s) = 0.0`
- **Radio UI (`web/index.html` + `src/radio_server.py`):**
  - 単一HTML + YouTube IFrame Player API（`ENDED` / `ERROR` フックによる無停止フォールバック、Rolling Prefetch Queue `N=3`、4操作UI、推薦理由1行表示）。

## Quickstart

```bash
pip install -r requirements.txt
```

### 1. 視聴履歴から `config/stations.yaml` を自動生成・更新（任意）

以下のいずれか（または両方）を配置して CLI を実行します（両方とも `.gitignore` 登録済み）：

- **YouTube Music 認証:** `ytmusicapi browser` 等で生成した `config/browser.json`（`get_history()` / `get_liked_songs()` を取得）
- **Google Takeout:** YouTube の視聴履歴 `data/watch-history.json`（通常 YouTube の音楽動画を自動抽出）

```bash
# 履歴ファイルを読み込み、MusicBrainzタグでクラスタリングして3局生成
python3 -m src.digger.history_importer --stations 3

# 履歴ファイル未配置の場合、仕様例のシード局に自動フォールバック
python3 -m src.digger.history_importer --offline
```

### 2. ローカルラジオサーバー起動

```bash
uvicorn src.radio_server:app --host 127.0.0.1 --port 8000 --reload
```

ブラウザで `http://127.0.0.1:8000/` を開くと、局切替・再生/ミュート・👍Good・👎BAD（即次曲）の4操作で連続再生が始まります。

### 3. テスト実行

```bash
pytest -v
```
