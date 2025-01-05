
train_env = get_vectorized_gym_env(base_env, args.env_type, args.num_train_ep, num_steps, featurize_fn=featurize_fn)
eval_env = get_vectorized_gym_env(base_env, args.env_type, args.num_eval_ep, num_steps, featurize_fn=featurize_fn)


oppo_pi = PPO(oppo_obs_dim, args.hidden_dim, act_dim, "cpu")
oppo_load_model_path = args.load_model_dir + args.oppo_pi_name
oppo_pi.load_params(oppo_load_model_path)


my_pi = PPO(my_obs_dim, args.hidden_dim, act_dim, args.device,
        num_steps, args.batch_size, args.actor_lr, args.critic_lr, args.entropy_coef,
        args.gamma, args.num_update_per_iter, args.clip_param, args.max_grad_norm, my_idxs)
my_load_model_path = args.load_model_dir + f"../seen_myinit/{args.oppo_pi_name.split('_')[0]}_myinit.pt"
my_pi.load_params(my_load_model_path)

best_sparse_rew_avg = -np.Inf

if args.env_type == "oc":
    annealer = LinearAnnealer(horizon=args.reward_shaping_episodes)

num_train_epoch = args.num_episodes // args.num_train_ep

for i in range(num_train_epoch):
    
    if args.env_type == "oc":
        reward_shaping_param = annealer.param_value(i * args.num_train_ep)
    else:
        reward_shaping_param = None

    ep_return = [np.zeros((args.num_train_ep)) for _ in range(len(oppo_idxs+my_idxs))]

    while True:
        act_index_n = [None for _ in range(len(oppo_idxs+my_idxs))]
        act_prob_n = [None for _ in range(len(oppo_idxs+my_idxs))]
        
        for j in oppo_idxs:
            _, oppo_act_idx, oppo_act_prob, _ = oppo_pi.select_action(obs_n[:, j])
            act_index_n[j] = oppo_act_idx
            act_prob_n[j] = oppo_act_prob

        for j in my_idxs:
            _, my_act_idx, my_act_prob, _ = my_pi.select_action(obs_n[:, j])
            act_index_n[j] = my_act_idx
            act_prob_n[j] = my_act_prob
        
        act_index_n = np.array(act_index_n).transpose(1,0)
        
        next_obs_dict, reward_n, done_n, info = train_env.step(act_index_n)
        
        done = any(done_n)
        
        for j in my_idxs:
            obs_n_j = obs_n[:, j]
            if obs_n_j.dtype == np.object_:
                obs_n_j = np.array(obs_n_j.tolist(), dtype=np.float32)
            ppo_trans = PPO_Transition(obs_n_j, act_index_n[:, j], act_prob_n[j], reward_n[:, j], next_obs_n[:, j])
            my_pi.store_transition(ppo_trans, j)

        for j in oppo_idxs+my_idxs:
            ep_return[j] += reward_n[:, j]
        
        obs_n = next_obs_n
        
        if done:
            break
        