"""
カテゴリ別完全分離の進化システム
各カテゴリを独立した集団として進化させる
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import boto3
from .llm_utils import LLMClient
from .evolver import PromptCandidate, DefenseCandidate, PromptEvolver, DefenseEvolver


class CategoryEvolutionManager:
    """カテゴリ別進化マネージャー"""
    
    def __init__(
        self,
        llm_client,
        evaluator,
        output_dir: Path,
        model_id: str = "mistral.mistral-large-2407-v1:0",
        max_workers: int = 5
    ):
        """
        Args:
            llm_client: boto3のBedrockクライアント
            evaluator: プロンプト評価器
            output_dir: 結果保存ディレクトリ
            model_id: 使用するLLMモデルID
            max_workers: 並行処理のワーカー数
        """
        self.attack_evolver = PromptEvolver(llm_client, model_id)
        self.defense_evolver = DefenseEvolver(llm_client, model_id)
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self._lock = threading.Lock()
        
        # カテゴリごとの履歴
        self.category_histories: Dict[str, List[List[PromptCandidate]]] = {}
        self.defense_history: List[DefenseCandidate] = []
    
    def run_category_evolution(
        self,
        category_populations: Dict[str, List[PromptCandidate]],
        initial_defense: DefenseCandidate,
        num_generations: int = 5,
        population_size_per_category: int = 3
    ) -> Tuple[Dict[str, List[PromptCandidate]], DefenseCandidate]:
        """
        カテゴリ別完全分離進化を実行
        
        Args:
            category_populations: カテゴリごとの初期集団
            initial_defense: 初期防御プロンプト
            num_generations: 世代数
            population_size_per_category: 各カテゴリの集団サイズ
            
        Returns:
            (カテゴリごとの最終集団, 最終防御プロンプト)
        """
        
        current_populations = category_populations
        current_defense = initial_defense
        
        # カテゴリリスト
        categories = list(category_populations.keys())
        
        for generation in range(num_generations):
            print(f"\n{'='*70}")
            print(f"Generation {generation + 1}/{num_generations}")
            print(f"{'='*70}")
            
            # 全カテゴリの評価と進化（並列処理）
            all_evaluated = []
            next_populations = {}
            
            print(f"\n並列処理開始 (ワーカー数: {self.max_workers})")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # カテゴリごとの評価タスクを投入
                future_to_category = {}
                for category in categories:
                    future = executor.submit(
                        self._process_category,
                        category,
                        current_populations[category],
                        current_defense,
                        population_size_per_category,
                        generation + 1
                    )
                    future_to_category[future] = category
                
                # 結果を収集
                for future in as_completed(future_to_category):
                    category = future_to_category[future]
                    try:
                        evaluated, next_pop = future.result()
                        next_populations[category] = next_pop
                        all_evaluated.extend(evaluated)
                        
                        # 進捗表示
                        avg_score = sum(a.score for a in evaluated) / len(evaluated) if evaluated else 0
                        best_score = max(a.score for a in evaluated) if evaluated else 0
                        print(f"\n【カテゴリ: {category}】完了  → 最高: {best_score:.1f}")
                    except Exception as e:
                        print(f"\n❌ カテゴリ {category} でエラー: {e}")
                        # エラー時はフォールバック
                        next_populations[category] = current_populations[category]
            
            # 防御プロンプトの進化（全カテゴリの評価結果を使用）
            print(f"\n【防御プロンプトの進化】")
            current_defense = self._evolve_defense(current_defense, all_evaluated)
            
            # 履歴保存
            for category in categories:
                if category not in self.category_histories:
                    self.category_histories[category] = []
                # 評価済み集団を取得
                evaluated_pop = [a for a in all_evaluated if category in a.category_scores]
                self.category_histories[category].append(evaluated_pop)
            
            self.defense_history.append(current_defense)
            
            # 世代結果の保存
            self._save_generation_results(generation + 1, next_populations, current_defense)
            
            # 次世代へ
            current_populations = next_populations
            
            # 全体の進捗表示
            print(f"\n世代 {generation + 1} 完了:")
            total_score = sum(a.score for a in all_evaluated)
            total_count = len(all_evaluated)
            print(f"  全体平均スコア: {total_score/total_count:.2f}")
            print(f"  防御ブロック率: {current_defense.avg_block_rate:.1%}")
        
        # 最終選抜: 各カテゴリから1つずつ選ぶ
        print(f"\n{'='*70}")
        print("最終提出用プロンプトの選抜（各カテゴリから1つ）")
        print(f"{'='*70}")
        
        final_attacks = self._select_one_from_each_category(current_populations)
        
        return current_populations, final_attacks, current_defense
    
    def _process_category(
        self,
        category: str,
        population: List[PromptCandidate],
        defense: DefenseCandidate,
        population_size: int,
        generation_num: int
    ) -> Tuple[List[PromptCandidate], List[PromptCandidate]]:
        """カテゴリの評価と進化を一括処理（並列実行用）"""
        
        # 評価
        evaluated = self._evaluate_category_attacks(population, defense, category)
        
        # 進化
        next_pop = self._evolve_category_population(
            evaluated,
            category,
            population_size,
            generation_num
        )
        
        return evaluated, next_pop
    
    def _evaluate_category_attacks(
        self,
        attacks: List[PromptCandidate],
        defense: DefenseCandidate,
        category: str
    ) -> List[PromptCandidate]:
        """カテゴリ内の攻撃プロンプト群を評価（並列実行対応）"""
        
        evaluated = []
        
        # 攻撃プロンプトの評価を並列化
        with ThreadPoolExecutor(max_workers=min(3, len(attacks))) as executor:
            future_to_attack = {}
            for attack in attacks:
                future = executor.submit(self._evaluate_single_attack, attack, defense, category)
                future_to_attack[future] = attack
            
            for future in as_completed(future_to_attack):
                attack = future_to_attack[future]
                try:
                    evaluated_attack = future.result()
                    evaluated.append(evaluated_attack)
                except Exception as e:
                    print(f"    ❌ {attack.prompt_id}: 評価エラー - {e}")
                    # エラー時はスコア0で追加
                    attack.score = 0
                    evaluated.append(attack)
        
        return evaluated
    
    def _evaluate_single_attack(
        self,
        attack: PromptCandidate,
        defense: DefenseCandidate,
        category: str
    ) -> PromptCandidate:
        """単一の攻撃プロンプトを評価"""
        
        # 評価器で評価
        result = self.evaluator.evaluate(attack.prompt_text, defense.defense_text)
        
        # 結果を反映
        attack.score = result.get("total_score", 0)
        attack.category_scores = result.get("category_scores", {})
        attack.stage_results = result
        
        with self._lock:
            print(f"    {attack.prompt_id}: Score={attack.score:.1f}, {result.get('block_reason')}")
        
        return attack
    
    def _evolve_category_population(
        self,
        evaluated_attacks: List[PromptCandidate],
        target_category: str,
        population_size: int,
        generation_num: int
    ) -> List[PromptCandidate]:
        """カテゴリ専用の集団を進化させる"""
        
        # スコアでソート
        sorted_attacks = sorted(evaluated_attacks, key=lambda x: x.score, reverse=True)
        
        next_generation = []
        
        # 1. エリート保持 (1個 = 33%)
        if sorted_attacks:
            elite = copy.deepcopy(sorted_attacks[0])
            elite.generation = generation_num
            elite.prompt_id = f"cat_{target_category}_gen{generation_num}_elite"
            next_generation.append(elite)
        
        # 2. 進化 (1個 = 33%)
        if len(sorted_attacks) > 0:
            parent = sorted_attacks[0]  # 最良個体から進化
            feedback = parent.stage_results
            
            evolved_text = self.attack_evolver.evolve_attack_prompt(
                parent,
                target_category,
                feedback
            )
            
            if evolved_text:
                new_attack = PromptCandidate(
                    prompt_id=f"cat_{target_category}_gen{generation_num}_evolved",
                    prompt_text=evolved_text,
                    generation=generation_num,
                    parent_id=parent.prompt_id
                )
                next_generation.append(new_attack)
        
        # 3. 変種 (1個 = 33%)
        if len(sorted_attacks) > 0:
            base = sorted_attacks[0]
            variants = self.attack_evolver.generate_diverse_attacks(base, num_variants=1)
            
            if variants and variants[0]:
                new_attack = PromptCandidate(
                    prompt_id=f"cat_{target_category}_gen{generation_num}_variant",
                    prompt_text=variants[0],
                    generation=generation_num,
                    parent_id=base.prompt_id
                )
                next_generation.append(new_attack)
        
        # 不足分を補完
        while len(next_generation) < population_size and sorted_attacks:
            idx = len(next_generation) % len(sorted_attacks)
            copy_attack = copy.deepcopy(sorted_attacks[idx])
            copy_attack.generation = generation_num
            copy_attack.prompt_id = f"cat_{target_category}_gen{generation_num}_copy{len(next_generation)}"
            next_generation.append(copy_attack)
        
        return next_generation[:population_size]
    
    def _evolve_defense(
        self,
        current_defense: DefenseCandidate,
        all_evaluated_attacks: List[PromptCandidate]
    ) -> DefenseCandidate:
        """防御プロンプトを進化させる（全カテゴリの攻撃に対応）"""
        
        # 攻撃の成功・失敗を分類
        successful_attacks = [a for a in all_evaluated_attacks if a.score >= 10]
        failed_attacks = [a for a in all_evaluated_attacks if a.score < 10]
        
        # ブロック率を計算
        block_rate = len(failed_attacks) / len(all_evaluated_attacks) if all_evaluated_attacks else 0
        
        # 進化
        evolved_text = self.defense_evolver.evolve_defense_prompt(
            current_defense,
            successful_attacks,
            failed_attacks
        )
        
        if evolved_text:
            new_defense = DefenseCandidate(
                defense_id=f"defense_gen{current_defense.generation + 1}",
                defense_text=evolved_text,
                generation=current_defense.generation + 1,
                avg_block_rate=block_rate,
                blocked_attacks=len(failed_attacks),
                total_attacks=len(all_evaluated_attacks),
                parent_id=current_defense.defense_id
            )
            return new_defense
        else:
            return current_defense
    
    def _select_one_from_each_category(
        self,
        category_populations: Dict[str, List[PromptCandidate]]
    ) -> List[PromptCandidate]:
        """各カテゴリから最良の1つを選抜"""
        
        all_candidates = []
        
        print("\n各カテゴリの最良個体:")
        for category, population in category_populations.items():
            # スコアでソート
            sorted_pop = sorted(population, key=lambda x: x.score, reverse=True)
            
            if sorted_pop:
                best = sorted_pop[0]
                all_candidates.append(best)
                print(f"  {category}: {best.prompt_id} (Score={best.score:.1f})")
        
        return all_candidates
    
    def select_final_5_for_submission(
        self,
        all_candidates: List[PromptCandidate]
    ) -> List[PromptCandidate]:
        """
        最終提出用5個を選抜（スコアとカテゴリ多様性を考慮、2000文字制限チェック付き）
        
        Args:
            all_candidates: 候補プロンプト群
            
        Returns:
            選抜された5個のプロンプト
        """
        
        print(f"\n{'='*70}")
        print("最終提出用5個の選抜")
        print(f"{'='*70}")
        
        # 2000文字以内の候補のみにフィルタ
        valid_candidates = [c for c in all_candidates if len(c.prompt_text) <= 2000]
        over_limit = [c for c in all_candidates if len(c.prompt_text) > 2000]
        if over_limit:
            print(f"\n⚠️ 2000文字超過のため除外: {len(over_limit)}個")
            for candidate in over_limit:
                print(f"  - {candidate.prompt_id}: {len(candidate.prompt_text)}文字 (超過: {len(candidate.prompt_text)-2000}文字)")
        if not valid_candidates:
            print("\n❌ エラー: 2000文字以内の候補がありません！")
            return []
        print(f"\n有効な候補: {len(valid_candidates)}個")
        
        # スコアでソート
        sorted_candidates = sorted(valid_candidates, key=lambda x: x.score, reverse=True)
        
        final_5 = []
        used_categories = set()
        
        print("\n選抜戦略: スコア優先 + カテゴリ多様性 + 2000文字制限遵守")
        
        # 1. 各カテゴリから最高スコア1つずつ選ぶ
        category_best = {}
        for candidate in sorted_candidates:
            for cat in candidate.category_scores.keys():
                if cat not in category_best:
                    category_best[cat] = candidate
        
        # カテゴリごとの最良をスコア順にソート
        category_ranked = sorted(
            category_best.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        # 重複除去して上位5個を選ぶ
        seen_ids = set()
        for candidate in category_ranked:
            if candidate.prompt_id not in seen_ids and len(final_5) < 5:
                final_5.append(candidate)
                seen_ids.add(candidate.prompt_id)
                # カテゴリを記録
                for cat in candidate.category_scores.keys():
                    used_categories.add(cat)
        
        # 2. まだ枠が余っていれば高スコア順で追加
        if len(final_5) < 5:
            for candidate in sorted_candidates:
                if candidate.prompt_id not in seen_ids:
                    final_5.append(candidate)
                    seen_ids.add(candidate.prompt_id)
                    used_categories.update(candidate.category_scores.keys())
                    if len(final_5) >= 5:
                        break
        
        # 結果表示
        print(f"\n選抜結果: {len(final_5)}個のプロンプト")
        all_categories = set()
        for i, candidate in enumerate(final_5, 1):
            categories = list(candidate.category_scores.keys())
            all_categories.update(categories)
            char_count = len(candidate.prompt_text)
            print(f"  {i}. {candidate.prompt_id}")
            print(f"     Score: {candidate.score:.1f}")
            print(f"     Categories: {categories}")
            print(f"     Stage: {candidate.stage_results.get('stage_reached', 0)}")
            print(f"     文字数: {char_count}/2000文字 {'✓' if char_count <= 2000 else '⚠️超過'}")
        
        print(f"\nカバーしたカテゴリ数: {len(all_categories)}/7")
        print(f"カバーしたカテゴリ: {sorted(all_categories)}")
        
        # 最終確認
        invalid_count = sum(1 for c in final_5 if len(c.prompt_text) > 2000)
        if invalid_count > 0:
            print(f"\n⚠️ 警告: {invalid_count}個のプロンプトが2000文字を超えています")
        else:
            print(f"\n✓ 全てのプロンプトが2000文字以内です")
        
        return final_5
    
    def save_submission_file(
        self,
        final_attacks: List[PromptCandidate],
        final_defense: DefenseCandidate,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        最終提出用CSVファイルを生成（sample_submission形式）
        
        Args:
            final_attacks: 最終5個の攻撃プロンプト
            final_defense: 最終防御プロンプト
            output_path: 出力先パス（Noneの場合はデフォルト）
            
        Returns:
            生成したCSVファイルのパス
        """
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir.parent.parent / "data" / "submission" / f"submission_category_evolved_{timestamp}.csv"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 提出用データを構築（sample_submission形式）
        submission_data = []
        
        # 5つの攻撃プロンプト
        for i, attack in enumerate(final_attacks[:5], 1):
            submission_data.append({
                "prompt_type": "attack",
                "prompt_id": f"attack_{i}",
                "prompt_text": attack.prompt_text
            })
        
        # 不足分を埋める（安全のため）
        while len(submission_data) < 5:
            idx = len(submission_data) + 1
            submission_data.append({
                "prompt_type": "attack",
                "prompt_id": f"attack_{idx}",
                "prompt_text": final_attacks[0].prompt_text if final_attacks else ""
            })
        
        # 防御プロンプト
        submission_data.append({
            "prompt_type": "defense",
            "prompt_id": "defense_1",
            "prompt_text": final_defense.defense_text
        })
        
        # CSV保存
        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        print(f"\n{'='*70}")
        print("提出用ファイル生成完了")
        print(f"{'='*70}")
        print(f"保存先: {output_path}")
        print(f"ファイルサイズ: {output_path.stat().st_size:,} bytes")
        print(f"\n内容:")
        print(f"  - 攻撃プロンプト: 5個 (attack_1 ~ attack_5)")
        print(f"  - 防御プロンプト: 1個 (defense_1)")
        
        # 統計情報
        total_chars = sum(len(attack.prompt_text) for attack in final_attacks[:5])
        avg_chars = total_chars / min(5, len(final_attacks)) if final_attacks else 0
        print(f"\n攻撃プロンプト統計:")
        print(f"  - 合計文字数: {total_chars:,}文字")
        print(f"  - 平均文字数: {avg_chars:.0f}文字")
        print(f"  - 防御文字数: {len(final_defense.defense_text):,}文字")
        
        # カテゴリ情報
        all_categories = set()
        for attack in final_attacks[:5]:
            all_categories.update(attack.category_scores.keys())
        print(f"\nカバーしたカテゴリ: {sorted(all_categories)}")
        
        # フォーマット確認
        print(f"\nフォーマット: sample_submission準拠")
        print(f"  - カラム: prompt_type, prompt_id, prompt_text")
        print(f"  - 行数: {len(submission_data)}行 (ヘッダー除く)")
        
        return output_path
    
    def _save_generation_results(
        self,
        generation: int,
        category_populations: Dict[str, List[PromptCandidate]],
        defense: DefenseCandidate
    ):
        """世代の結果を保存"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # カテゴリごとに保存
        for category, population in category_populations.items():
            cat_file = self.output_dir / f"gen{generation:02d}_{category}_{timestamp}.json"
            with open(cat_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [asdict(a) for a in population],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        
        # 防御プロンプトの保存
        defense_file = self.output_dir / f"gen{generation:02d}_defense_{timestamp}.json"
        with open(defense_file, 'w', encoding='utf-8') as f:
            json.dump(
                asdict(defense),
                f,
                ensure_ascii=False,
                indent=2
            )
        
        print(f"\n世代 {generation} の結果を保存: {self.output_dir}")


def load_initial_populations_from_variants(
    variant_data: Dict[str, List[Dict]],
    population_size_per_category: int = 3
) -> Dict[str, List[PromptCandidate]]:
    """
    prompt_variants.pyからカテゴリ別初期集団を生成
    
    Args:
        variant_data: カテゴリ名 -> プロンプトリストの辞書
        population_size_per_category: 各カテゴリの集団サイズ
        
    Returns:
        カテゴリごとの初期PromptCandidate集団
    """
    
    category_populations = {}
    
    for category, variants in variant_data.items():
        population = []
        
        # 各カテゴリから指定数を選択
        for i, variant in enumerate(variants[:population_size_per_category]):
            candidate = PromptCandidate(
                prompt_id=f"cat_{category}_init_{i+1}",
                prompt_text=variant["text"],
                generation=0,
                category_scores={category: 0}  # 初期は未評価
            )
            population.append(candidate)
        
        category_populations[category] = population
    
    return category_populations
