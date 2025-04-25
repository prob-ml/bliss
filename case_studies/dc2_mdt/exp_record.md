
(due to a bug in code, the previous exp about mdt are all invalid)

### 04-12 (mdt)

1. train a basic mdt

### 04-12 (simple_net)

2. train a basic simple net
3. pred velocity


### 04-14 (great_lake)

1. mdt
(these simple_nets are trained to predict velocity, so the loss is high)
2. simple_net
3. simple_net_speed
4. simple_ar_net
5. simple_cond_true_net

### 04-14 (simple_net)

(these simple_nets are trained to predict velocity, so the loss is high)
1. train a basic simple net (batch_size: 128)
2. train a basic simple net (batch_size: 64)

3. train a basic simple net (batch_size: 128) pred noise


### 04-15 (simple_net)

1. train a basic simple net (batch_size: 64) pred noise

### 04-19 (m2 mdt)

1. run mdt on m2

### 04-20 (m2 simple_net)

1. run simple net on m2

### 04-20 (m2 ori_bliss)

2. run ori bliss on m2

### 04-20 (m2 cond_true_bliss)

3. run on m2

### 04-21

1. (m2 mdt) train without learning sigma
2. (m2 mdt) train without learning sigma; use speed sampler

### 04-34

1. (m2 mdt rml) beta: 0.01; lr: 3e-4
2. (m2 mdt rml) beta: 0.5; lr: 1e-3
3. (m2 mdt rml) beta: 0.5; lr: 3e-4
4. (m2 mdt rml) beta: 0.5; lr: 3e-4; use vmap
