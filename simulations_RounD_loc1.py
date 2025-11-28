import numpy as np
import pandas as pd
from experiment import Experiment


# Draw latex figure
Experiment_name = 'RounD_1'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# # Select the datasets
Data_sets = [{'scenario': 'RounD_round_about', 'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.2, 'num_timesteps_in': (15, 15), 'num_timesteps_out': (25, 25)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'Location_split', 'repetition':'1'}]

# Select the models to be trained
Models = []

for seed in range(5):
	Models.append({
		'model': 'GMoP_meszaros_multiTask',
		'kwargs': {'social_enc_type': 'Classified',
					'crossing_task': False,
					'hyp_crossing_task': False,
					'closeness_task': False,
					'batch_size': 128,
					'interaction_thresh': 0.3,
					'use_img': False,
					'seed': seed}})

	Models.append({
		'model': 'GMoP_meszaros_jointPreTrainInter',
		'kwargs': {'social_enc_type': 'Classified',
					'crossing_task': True,
					'batch_size': 128,
					'interaction_thresh': 0.3,
					'use_img': False,
					'seed': seed}})
					
	Models.append({
		'model': 'GMoP_meszaros_jointPreTrainInter',
		'kwargs': {'social_enc_type': 'Classified',
					'hyp_crossing_task': True,
					'batch_size': 128,
					'interaction_thresh': 0.3,
					'use_img': False,
					'seed': seed}})
	
	Models.append({
	'model': 'GMoP_meszaros_multiTask_Euclidean',
	'kwargs': {'social_enc_type': 'Classified',
				'batch_size': 128,
				'interaction_thresh': 0.3,
				'use_img': False,
				'seed': seed}})
	
	Models.append({
		'model': 'GMoP_meszaros_multiTask_indep',
		'kwargs': {'social_enc_type': 'Classified',
					'crossing_task': False,
					'hyp_crossing_task': False,
					'closeness_task': False,
					'batch_size': 128,
					'interaction_thresh': 0.3,
					'use_img': False,
					'seed': seed}})
	
	Models.append({
		'model': 'GMoP_meszaros_jointPreTrainInter_Flipped',
		'kwargs': {'social_enc_type': 'Classified',
					'crossing_task': True,
					'batch_size': 128,
					'interaction_thresh': 0.3,
					'use_img': False,
					'seed': seed}})
					
	Models.append({
		'model': 'GMoP_meszaros_jointPreTrainInter_Flipped',
		'kwargs': {'social_enc_type': 'Classified',
					'hyp_crossing_task': True,
					'batch_size': 128,
					'interaction_thresh': 0.3,
					'use_img': False,
					'seed': seed}})
	
	Models.append({
		'model': 'fjmp_rowe',
		'kwargs': {'max_epochs': 36,
					'batch_size': 64,
					'num_joint_modes': 6,
					'seed': seed}})
		
	Models.append({
		'model': 'autobot_girgis',
		'kwargs': {'seed': seed}})
	
	Models.append({
		'model': 'adapt_aydemir',
		'kwargs': {'seed': seed}})
		
    

# Select the metrics to be used
Metrics = [{'metric': 'minADE_joint', 'kwargs': {'num_preds': 6}}, {'metric': 'minFDE_joint', 'kwargs': {'num_preds': 6}}, 'KDE_NLL_joint']

new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 100

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_time = True

# determine if the upper bound for n_O should be enforced, or if prediction can be made without
# underlying output data (might cause training problems)
enforce_num_timesteps_out = True

# Determine if the useless prediction (i.e, prediction you cannot act anymore)
# should be exclude from the dataset
exclude_post_crit = True

# Decide wether missing position in trajectory data can be extrapolated
allow_extrapolation = False

# Use all available agents for predictions
agents_to_predict = 'all'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = False

# Determine if predictions should be saved
save_predictions = True

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_time, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, 
                              save_predictions, evaluate_on_train_set)


#%% Run experiment
new_experiment.run() 

# Load results
Results = new_experiment.load_results(plot_if_possible = False)

np.set_printoptions(precision=3, suppress=True, linewidth=300)
R = Results.squeeze()
 
# Take mean over seeds
M = pd.DataFrame([pd.Series(np.array([m['model']] + list(m['kwargs'].values())), index =(['model'] + list(m['kwargs'].keys()))) for m in Models])
M = M.iloc[(M.seed == '0').to_numpy() | (M.seed == 0).to_numpy()]
M = M.drop(columns=['batch_size', 'hs_rnn', 'n_layers_rnn', 'fut_enc_sz', 'scene_encoding_size', 'beta_noise', 'gamma_noise', 'alpha', 'fut_ae_epochs',
                    'fut_ae_lr_decay', 'fut_ae_lr','fut_ae_wd', 'flow_epochs', 'flow_lr', 'flow_lr_decay', 'flow_wd', 'vary_input_length', 'pos_loss', 'seed',
                    'interaction_thresh', 'multiTask_lossWeight', 'use_img', 'use_graph', 'social_enc_type', 'es_gnn', 'cross_dist', 'cross_angle', 'preprocess', 
					'val_workers','workers', 'num_agenttypes','num_scales','map2actor_dist','actor2actor_dist', 'lr_step', 'input_size', 'prediction_steps' ,
					'observation_steps', 'dataset', 'switch_lr_1', 'switch_lr_2', 'no_agenttype_encoder', 'train_all', 'supervise_vehicles', 'scheduled_sampling', 
					'teacher_forcing', 'focal_loss', 'gamma', 'weight_0', 'weight_1', 'weight_2', 'proposal_header', 'two_stage_training', 'training_stage', 'ig',
					'n_l2a_layers', 'n_a2a_layers', 'proposal_coef', 'rel_coef', 'decoder', 'num_heads', 'learned_relation_header', 'n_mapnet_layers', 'num_edge_types', 
					'h_dim', 'num_proposals', 'lr', 'obs_encoding_size', 'iCL_hdim', 'max_epochs', 'hidden_size', 'num_encoder_layers', 'num_decoder_layers', 
					'tx_hidden_size', 'tx_num_heads', 'dropout', 'entropy_weight', 'kl_weight', 'use_FDEADE_aux_loss', 'predict_yaw', 'k_attr', 'map_attr', 'learning_rate',
					'learning_rate_sched', 'adam_epsilon', 'optimizer', 'scheduler', 'train_batch_size', 'eval_batch_size', 'grad_clip_norm', 'map_range', 
					'point_sampled_interval', 'num_points_each_polyline', 'vector_break_dist_thresh', 'epoch', 'layer_num', 'multi_agent', 'static_agent_drop',
					'scaling', 'use_checkpoint', 'max_distance', 'model_name', 'closeness_task'])

 
useful  = np.isfinite(R).all(1)
m = M.to_numpy()
 
m = np.tile(m, (5, 1))
m = m[:,1:]
 
r = R[useful]
m = m[useful]
  
 
# Take mean over seeds
R = R.reshape(5, -1, len(Metrics))

Std = np.nanstd(R, axis=0)
Mean = np.nanmean(R, axis=0)
 
useless = np.isnan(Mean).all(1)
Mean = Mean[~useless]
M = M.iloc[~useless]
Std = Std[~useless]

Metrics = [{'metric': 'minADE_joint', 'kwargs': {'num_preds': 6}}, {'metric': 'minFDE_joint', 'kwargs': {'num_preds': 6}}, 'KDE_NLL_joint']
 
print('Correlation between metrics')
corr = np.corrcoef(Mean, rowvar=False)
print(corr)
M_std = M.copy()
M['minADE6_joint'] = Mean[:, 0]
M['minFDE6_joint'] = Mean[:, 1]
M['KDE_NLL_joint'] = Mean[:, 2]
 
M.to_excel("RounD_Results_loc1.xlsx")
 
M_std['minADE6_joint'] = Std[:, 0]
M_std['minFDE6_joint'] = Std[:, 1]
M_std['KDE_NLL_joint'] = Std[:, 2]

M_std.to_excel("RounD_Results_loc1_std.xlsx")