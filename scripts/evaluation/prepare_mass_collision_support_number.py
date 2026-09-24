"""Freeze nested K=1/2/4 support sets and a common disjoint query set."""
import json
from pathlib import Path
import yaml

root=Path('results/mass_collision/noleak_highmass2x_support_k1_k2_k4_all_model_predictions_v1')
meta=Path('data/mass_collision_noleak_all30_mainview_bwm_full61_20260828/train.jsonl')
rows=[json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
reference=Path('results/mass_collision/noleak_highmass2x_grid_id5_ood5_k1_balanced_visible_or_min_action_v1/methods/ours_action8_highmass2x/step_4300/seed_20260827/same_mass_other_actions')
base=yaml.safe_load(Path('configs/evaluation/action_tasks/mass_collision_noleak_ours_action8_highmass2x_step4300_id5_ood5_minimum_long_v1.yaml').read_text())
groups=[]
for domain in ('id','ood'):
 for item in json.loads((reference/f'transfer_plan_{domain}.json').read_text()):
  source=int(item['source_index']);pool=sorted({source,*map(int,item['target_indices'])},key=lambda i:int(rows[i]['speed_index']))
  assert len(pool)==9
  assert len({float(rows[i]['friction_mu']) for i in pool})==1
  selected=[source]
  while len(selected)<4:
   candidates=[i for i in pool if i not in selected]
   selected.append(max(candidates,key=lambda i:(min(abs(int(rows[i]['speed_index'])-int(rows[j]['speed_index'])) for j in selected),-int(rows[i]['speed_index']))))
  groups.append({'source_index':source,'domain':domain,'environment_id':rows[source]['target_mass_index'],'friction_mu':rows[source]['friction_mu'],'all_action_indices':pool,'nested_support_indices':selected,'common_query_indices':[i for i in pool if i not in selected]})
assert len(groups)==10
for k in (1,2,4):
 out=root/'methods/ours_action8_highmass2x'/f'k{k}'/'step_4300/seed_20260827'
 out.mkdir(parents=True,exist_ok=True)
 plan=[];support_rows=[]
 for g in groups:
  supports=g['nested_support_indices'][:k]
  assert not set(supports)&set(g['common_query_indices'])
  plan.append({**g,'support_indices':supports,'query_indices':g['common_query_indices'],'target_indices':g['all_action_indices']})
  support_rows.extend(rows[i] for i in supports)
 (out/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
 (out/'support_metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in support_rows))
 transfer=out/'transfer';transfer.mkdir(exist_ok=True)
 for mode in ('action','video'):
  c={**base,'method_name':f'ours_k{k}','support_size':k,'minimum_actions':9,'use_model_predictions_for_support':True,'output_dir':str(out/'action_evaluation'),'transfer_plans':[{'path':str(transfer/f'{mode}_plan.json'),'domain_by_source':{g['source_index']:g['domain'] for g in groups}}]}
  if mode=='video':
   c['output_dir']=str(out/'all_action_video_metrics/action_evaluation')
   c['video_metrics_include_support_predictions']=True
  (out/f'{mode}_evaluation.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
(root/'protocol.json').write_text(json.dumps({'checkpoint':'outputs/mass20of30_collision_noleak_mainview_c32_oldrandom_8action_highmass2x_s1_108592/step-4300.safetensors','support_sizes':[1,2,4],'groups':groups,'additional_support_selection':'greedy_farthest_speed_index_lower_speed_tiebreak','support_loss_reduction':'mean','inner_steps':40,'inner_lr_schedule':'3.0:10,1.5:10,0.5:10,0.15:10','spatial_loss_mode':'none','action_decisions':'model predictions for all nine actions, including support actions; GT only scores decisions','video_prediction_count_per_environment':9,'video_metrics_include_support_predictions':True,'video_metrics_are_disjoint_query_only':False,'common_disjoint_query_count_per_environment':5,'selection_uses_additional_support_gt_outcomes':False},indent=2)+'\n')
print(root)
