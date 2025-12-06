# プロンプト自動進化システム - セットアップ手順

## 1. 環境セットアップ

### Python環境
```bash
# 仮想環境を作成（推奨）
python -m venv venv
venv\Scripts\activate  # Windows
# または
source venv/bin/activate  # Mac/Linux

# 依存パッケージのインストール
pip install -r requirements.txt
```

### AWS認証設定

AWS Bedrockを使用するため、認証情報が必要です。

**方法1: AWS CLIで設定**
```bash
aws configure
# AWS Access Key ID: your_key
# AWS Secret Access Key: your_secret
# Default region name: us-east-1
# Default output format: json
```

**方法2: 環境変数**
```bash
# Windows (cmd)
set AWS_ACCESS_KEY_ID=your_key
set AWS_SECRET_ACCESS_KEY=your_secret
set AWS_DEFAULT_REGION=us-east-1

# Windows (PowerShell)
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"
$env:AWS_DEFAULT_REGION="us-east-1"

# Mac/Linux
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

**方法3: .envファイル（推奨）**
```bash
# .envファイルを作成
echo AWS_ACCESS_KEY_ID=your_key >> .env
echo AWS_SECRET_ACCESS_KEY=your_secret >> .env
echo AWS_DEFAULT_REGION=us-east-1 >> .env
```

### Bedrockモデルアクセス

AWS コンソールで Bedrock のモデルアクセスを有効化:
1. AWS コンソール → Bedrock
2. 左メニュー「Model access」
3. Claude 3.5 Sonnetを有効化

## 2. クイックスタート

### 最小限のテスト

```bash
# 高速評価モードで2世代だけ進化
python src/run_evolution.py --fast --generations 2 --population 3
```

実行時間: 約5-10分

### 通常実行

```bash
# 5世代進化（推奨設定）
python src/run_evolution.py --source strategic --generations 5 --population 5
```

実行時間: 約30-60分（LLM呼び出し回数により変動）

## 3. 実行例

### 例1: baselineから進化

```bash
python src/run_evolution.py \
    --source baseline \
    --generations 3 \
    --population 5
```

### 例2: strategicから高速テスト

```bash
python src/run_evolution.py \
    --source strategic \
    --fast \
    --generations 2 \
    --population 3 \
    --output-dir data/interim/test_run
```

### 例3: 本番実行（長時間）

```bash
python src/run_evolution.py \
    --source strategic \
    --generations 10 \
    --population 10
```

## 4. 出力の確認

### 世代ごとの結果

```bash
# 結果ファイルを確認
dir data\interim\evolution\  # Windows
ls data/interim/evolution/   # Mac/Linux
```

各世代のファイル:
- `generation_01_attacks_*.json`: 攻撃プロンプト群
- `generation_01_defense_*.json`: 防御プロンプト

### 提出ファイル

```bash
# 最終提出ファイルを確認
dir data\submission\submission_evolved_*.csv
```

このCSVファイルをコンペに提出できます。

## 5. 結果の分析

### Pythonで分析

```python
import json
import pandas as pd

# 最終世代の攻撃プロンプトを読み込み
with open('data/interim/evolution/generation_05_attacks_*.json', 'r') as f:
    attacks = json.load(f)

# スコアでソート
sorted_attacks = sorted(attacks, key=lambda x: x['score'], reverse=True)

# 上位3件を表示
for i, attack in enumerate(sorted_attacks[:3], 1):
    print(f"\n{i}位:")
    print(f"  スコア: {attack['score']}")
    print(f"  カテゴリ: {attack['category_scores']}")
    print(f"  テキスト: {attack['prompt_text'][:200]}...")
```

### ログの確認

進化過程のログを確認:
```bash
# 世代ごとのスコア推移を確認
grep "世代.*完了" logs/*.log
```

## 6. トラブルシューティング

### boto3が見つからない

```bash
pip install boto3
```

### AWS認証エラー

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

→ 「AWS認証設定」セクションを参照

### Bedrockアクセス拒否

```
AccessDeniedException: Could not access model
```

→ AWSコンソールでBedrockのモデルアクセスを有効化

### メモリ不足

大規模実行時にメモリ不足になる場合:
```bash
# 個体数を減らす
python src/run_evolution.py --population 3

# または世代数を減らす
python src/run_evolution.py --generations 3
```

### LLMレート制限

呼び出し頻度が高すぎる場合、エラーが出ることがあります:
```
ThrottlingException: Rate exceeded
```

→ しばらく待ってから再実行

## 7. カスタマイズ

### 評価戦略の変更

`src/evaluator.py` を編集して独自の評価ロジックを実装:

```python
class MyEvaluator(PromptEvaluator):
    def _stage1_prompt_safety_check(self, prompt: str) -> bool:
        # カスタム評価ロジック
        if "特定のキーワード" in prompt:
            return False
        return True
```

### 進化戦略の変更

`src/evolution.py` のメタプロンプトを編集:

```python
def _build_attack_evolution_prompt(self, ...):
    # 進化の指示を変更
    prompt = f"""あなたは...
    
    【独自の改善方針】
    1. ...
    2. ...
    """
    return prompt
```

## 8. パフォーマンスチューニング

### 高速化のヒント

1. **Fast評価モード**: 開発時は `--fast` を使用
2. **並列化**: 将来の拡張で評価を並列実行
3. **キャッシング**: 同じプロンプトの再評価を避ける

### コスト削減

1. **世代数を減らす**: `--generations 3`
2. **個体数を減らす**: `--population 3`
3. **Fast評価**: LLM呼び出しを最小化

## 9. 次のステップ

基本的な使い方をマスターしたら:

1. **戦略の調整**: `.github/copilot-instructions.md` の戦略表を参考に
2. **カスタム評価**: 独自の評価基準を実装
3. **アンサンブル**: 複数の進化実行結果を組み合わせ
4. **A/Bテスト**: 異なる初期値・パラメータで比較

## 10. FAQ

**Q: 何世代進化させるべき？**
A: 5-10世代が推奨。それ以上は過学習のリスク。

**Q: Fast評価と通常評価の違いは？**
A: Fastはキーワードベースの簡易評価。通常はLLMを使った詳細評価。

**Q: エラーで中断した場合は？**
A: 世代ごとに結果が保存されているので、そこから手動で継続可能。

**Q: 提出ファイルの選び方は？**
A: 複数の実行結果を比較し、最もスコアが高いものを選択。

**Q: コストはどのくらい？**
A: 5世代×5個体で約$5-10（Bedrockの料金）。Fast評価で削減可能。
