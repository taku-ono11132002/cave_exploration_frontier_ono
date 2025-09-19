# Cave Exploration (Frontier-best + Self-best)

## 概要
Y字型洞窟を3機ドローンでフロンティア探索する簡易シミュレータである。  
割り当ては次のルールである：
1. 各フロンティアに対して、そのフロンティアで **最良スコア** のドローンをまず確保（担当者）。
2. 残りのドローンは **自分にとって最良** のフロンティアへ向かう（重複可）。

## 実行方法
```bash
# Windows (PowerShell / cmd のどちらでも)
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

出力先: `outputs/` に MP4 と PNG を保存する。

## 依存関係
- numpy
- matplotlib
- imageio-ffmpeg  (ffmpeg バイナリ同梱・自動利用)

## フォルダ構成とファイル概要

```
.
├── main.py               # メインスクリプト
├── requirements.txt      # 依存ライブラリ
├── outputs/              # シミュレーション結果の出力先
│   ├── (動画ファイル).mp4
│   └── (画像ファイル).png
├── src/                  # ソースコード
│   ├── sim.py            # シミュレーション本体
│   ├── slam.py           # SLAM（自己位置推定と地図作成）
│   ├── frontier.py       # 未探索領域（フロンティア）の検出
│   ├── planner.py        # 経路計画とコスト計算
│   ├── assign.py         # フロンティアの割り当て戦略
│   └── animate.py        # 結果の動画・画像生成
└── README.md             # このファイル
```

### ファイル概要

- **`main.py`**:
  シミュレーションを実行し、結果を `outputs/` フォルダに動画(mp4)と最終状態の画像(png)として保存するエントリーポイントです。

- **`src/sim.py`**:
  シミュレーションのメインループを管理します。洞窟環境の生成、ロボットの状態更新、センサー情報の処理、探査戦略の実行など、シミュレーション全体の流れを制御します。

- **`src/slam.py`**:
  SLAM (Simultaneous Localization and Mapping) に関連する機能を提供します。占有格子地図 (`SLAMMap`) の管理や、LIDARセンサーによる地図の更新処理を担います。

- **`src/frontier.py`**:
  地図データから未探索領域（フロンティア）を検出し、それらをクラスタリングして探査対象の候補を生成します。

- **`src/planner.py`**:
  探査計画を立てるための計算を行います。A*アルゴリズムによる経路探索や、各ロボットから各フロンティアへの到達コストを計算する機能が含まれます。

- **`src/assign.py`**:
  計算されたコストに基づき、どのロボットがどのフロンティアを探査するかを決定する割り当て戦略を実装しています。

- **`src/animate.py`**:
  シミュレーションの各ステップの状態をフレームとして受け取り、matplotlibを用いてアニメーション動画やスナップショット画像を生成・保存します。

- **`outputs/`**:
  `main.py` を実行した際に生成される動画や画像が保存されるディレクトリです。
