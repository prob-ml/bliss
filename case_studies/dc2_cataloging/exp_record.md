## DC2 Experiment Record

### 05-25

1. thresholds: [-0.38, -0.1, -0.03, 0.008, 0.06, 0.18, 0.74]; scale: 1,000

### 05-26


1. thresholds: [-0.38, -0.1, -0.03, 0.008, 0.06, 0.18, 0.74]; scale: 10,000; using 2 GPUs

2. thresholds: [-0.1, -0.03, 0.008, 0.06, 0.18]; scale: 1000

3. thresholds: [-0.38, -0.1, -0.03, -0.014, 0.008, 0.034, 0.06, 0.18, 0.74]; scale: 10,000

(decide to use the setting 05-26-1 as standard setting)

### 05-27/28

it gives strange results but recovers when rebooting the server

### 05-29

1. run the experiment using the code and data in master branch

(use the filter function filter_tile_catalog_by_flux to filter target catalog and predicted catalog at threshold 100)

2. add dynamic asinh; use the setting of 05-26-1; run experiment on previous data using 4 GPUs

3. add dynamic asinh; use the setting of 05-26-1; run experiment on current data using 2 GPUs;

(write a test and find that the new DC2 class can generate the same data as previous DC2 does)

4. add dynamic asinh; use the setting of 05-26-1; add batch norm to process input data

(decide to use dynamic asinh)

5. add CLAHE; using 2 GPUs;

(decide to add CLAHE; we will use 2 GPUs by default)

6. change the scale to 1000

(decide to use thresholds [-0.38, -0.1, -0.03, 0.008, 0.06, 0.18, 0.738] and scale 1,000 as default setting)

### 05-30

1. fix the expand() memory bug in ImageNormalizer using clone()

2. disable galaxy property loss

(decide to disable galaxy_property_loss)

### 06-01

1. process asinh params using conv and average pooling

```python
        self.asinh_param_process = nn.Sequential(
                ConvBlock(2, 64, kernel_size=3, padding=1),
                ConvBlock(64, 64, kernel_size=3, padding=1),
                ConvBlock(64, 128, kernel_size=3, padding=1, stride=2),
                ConvBlock(128, 128, kernel_size=3, padding=1),
                ConvBlock(128, num_features, kernel_size=3, padding=1, stride=2),
            )

    def forward(self, x, asinh_params):
        if asinh_params is not None:
            processed_asinh_params = self.asinh_param_process(asinh_params.unsqueeze(0))
            # global averge pooling
            processed_asinh_params = processed_asinh_params.mean(dim=[2, 3], keepdim=True)
        x = self.preprocess3d(x).squeeze(2)
        x = self.backbone(x)
        return x * processed_asinh_params if asinh_params is not None else x
```

2. process asinh params using conv and max pooling

```python
processed_asinh_params, _ = processed_asinh_params.view(1, self.num_features, -1).max(dim=2, keepdim=True)
```

3. process asinh params using C3 and max pooling

```python
        self.asinh_param_process = C3(2, num_features, n=2)

    def forward(self, x, asinh_params):
        if asinh_params is not None:
            processed_asinh_params = self.asinh_param_process(asinh_params.unsqueeze(0))
            # max pooling
            processed_asinh_params, _ = processed_asinh_params.view(1, self.num_features, -1).max(dim=2, keepdim=True)
            processed_asinh_params = processed_asinh_params.unsqueeze(-1)
        x = self.preprocess3d(x).squeeze(2)
        x = self.backbone(x)
        return x * processed_asinh_params if asinh_params is not None else x
```

4. process asinh params using C3 + Linear and max pooling

```python
        self.asinh_param_process = C3(2, num_features, n=2)
        self.asinh_param_post_process = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x, asinh_params):
        if asinh_params is not None:
            processed_asinh_params = self.asinh_param_process(asinh_params.unsqueeze(0))
            # max pooling
            processed_asinh_params, _ = processed_asinh_params.view(1, self.num_features, -1).max(dim=2)
            processed_asinh_params = self.asinh_param_post_process(processed_asinh_params)
            processed_asinh_params = processed_asinh_params.unsqueeze(-1).unsqueeze(-1)
        x = self.preprocess3d(x).squeeze(2)
        x = self.backbone(x)
        return x * processed_asinh_params if asinh_params is not None else x
```

5. process asinh params using C3 + linear + groupnorm and max pooling

```python
        self.asinh_param_process = nn.Sequential(
            C3(2, num_features, n=2),
            nn.GroupNorm(8, num_features),
            nn.SiLU(),
        )
        self.asinh_param_post_process = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.SiLU(),
            nn.Linear(num_features // 2, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x, asinh_params):
        if asinh_params is not None:
            asinh_params_mean = asinh_params.mean(dim=2, keepdim=True)
            asinh_params_var = asinh_params.var(dim=2, keepdim=True)
            processed_asinh_params = (asinh_params - asinh_params_mean) / torch.sqrt(asinh_params_var + 1e-5)
            processed_asinh_params = self.asinh_param_process(asinh_params.unsqueeze(0)) # bug
            # max pooling
            processed_asinh_params, _ = processed_asinh_params.view(1, self.num_features, -1).max(dim=2)
            processed_asinh_params = self.asinh_param_post_process(processed_asinh_params)
            processed_asinh_params = processed_asinh_params.unsqueeze(-1).unsqueeze(-1)
        x = self.preprocess3d(x).squeeze(2)
        x = self.backbone(x)
        return x * processed_asinh_params if asinh_params is not None else x
```


### 06-02

1. process asinh params using transformer modules

```python
        self.asinh_param_encoder = nn.Sequential(
            C3(2, num_features, n=3),
            nn.Sequential(*[TransformerBlock(num_features) for _ in range(2)])
        )
        self.x_encoder = nn.Sequential(*[TransformerBlock(num_features) for _ in range(2)])
        self.merge_block = OutputTransformerBlock(num_features)

    def forward(self, x, asinh_params):
        preprocessed_x = self.preprocess3d(x).squeeze(2)
        backbone_x = self.backbone(preprocessed_x)

        if asinh_params is not None:
            asinh_params_mean = asinh_params.mean(dim=2, keepdim=True)
            asinh_params_var = asinh_params.var(dim=2, keepdim=True)
            normalized_asinh_params = (asinh_params - asinh_params_mean) / torch.sqrt(asinh_params_var + 1e-5)
            encoded_asinh_params = self.asinh_param_encoder(normalized_asinh_params.unsqueeze(0))

            encoded_backbone_x = self.x_encoder(backbone_x)
            attn_backbone_x = self.merge_block(encoded_backbone_x, encoded_asinh_params)

        return backbone_x + attn_backbone_x if asinh_params is not None else backbone_x
```

2. process asinh params using transformer modules, and let C3 use groupnorm

```python
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, use_group_norm=False):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, padding=0, use_group_norm=use_group_norm)
        self.cv2 = ConvBlock(c1, ch, kernel_size=1, padding=0, use_group_norm=use_group_norm)
        self.cv3 = ConvBlock(2 * ch, c2, kernel_size=1, padding=0, use_group_norm=use_group_norm)
        self.m = nn.Sequential(*(Bottleneck(ch, ch, shortcut, e=1.0, use_group_norm=use_group_norm) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

3. process asinh params using transformer modules, let C3 use groupnorm, and change adaptive_avg_pool2d to adaptive_max_pool2d

```python
y_q = y_q.reshape(y_b, self.num_heads, -1, y_h * y_w)
x_k = x_k.reshape(x_b, self.num_heads, -1, x_h * x_w)
x_v = x_v.reshape(x_b, self.num_heads, -1, x_h * x_w)
y_q, x_k = F.normalize(y_q, dim=-1), F.normalize(x_k, dim=-1)
x_k = F.adaptive_max_pool2d(x_k, output_size=(None, y_q.shape[-1]))
```

(adaptive_max_pool2d is not good, we still use adaptive_avg_pool2d)

4. failed exp

5. use a new architecture to preprocess asinh params

```python
class AsinhParamsPreprocess(nn.Module):
    def __init__(self, in_channels, use_group_norm=True):
        super().__init__()

        self.pad_input = nn.ZeroPad2d(padding=(0, 1, 1, 1))
        self.net_ml = nn.ModuleList([
            C3(in_channels, 64, n=3, use_group_norm=use_group_norm),
            ConvBlock(64, 64, use_group_norm=use_group_norm),
            ConvBlock(64, 128, stride=2, use_group_norm=use_group_norm),
            C3(128, 128, n=3, shortcut=False, use_group_norm=use_group_norm),
            ConvBlock(128, 128, kernel_size=1, padding=0, use_group_norm=use_group_norm),
            nn.Upsample(scale_factor=2, mode="nearest"),
        ])

    def forward(self, x):
        x = self.pad_input(x)
        save_lst = []
        for i, m in enumerate(self.net_ml):
            x = m(x)
            if i in {0, 1, 5}:
                save_lst.append(x)
            if i == 5:
                x = torch.cat(save_lst, dim=1)
        return x

    ...

        self.asinh_param_preprocess = AsinhParamsPreprocess(in_channels=2)
        self.asinh_param_encoder = nn.Sequential(*[TransformerBlock(num_features) for _ in range(2)])
        self.x_encoder = nn.Sequential(*[TransformerBlock(num_features) for _ in range(2)])
        self.merge_block = OutputTransformerBlock(num_features)

    def forward(self, x, asinh_params):
        preprocessed_x = self.preprocess3d(x).squeeze(2)
        backbone_x = self.backbone(preprocessed_x)

        if asinh_params is not None:
            asinh_params_mean = asinh_params.mean(dim=2, keepdim=True)
            asinh_params_var = asinh_params.var(dim=2, keepdim=True)
            normalized_asinh_params = (asinh_params - asinh_params_mean) / torch.sqrt(asinh_params_var + 1e-5)
            preprocessed_asinh_params = self.asinh_param_preprocess(normalized_asinh_params.unsqueeze(0))
            encoded_asinh_params = self.asinh_param_encoder(preprocessed_asinh_params)

            encoded_backbone_x = self.x_encoder(backbone_x)
            attn_backbone_x = self.merge_block(encoded_backbone_x, encoded_asinh_params)

        return backbone_x + attn_backbone_x if asinh_params is not None else backbone_x

```

6. use a new method to process asinh params

```python
        if self.use_asinh:
            self.asinh_preprocess = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 1, 1, 1)),
                C3(2, nch_hidden_for_asinh_params, n=4, use_group_norm=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvBlock(nch_hidden_for_asinh_params, nch_hidden_for_asinh_params, use_group_norm=True),
                nn.Upsample(scale_factor=5, mode="nearest"),
                ConvBlock(nch_hidden_for_asinh_params, nch_hidden_for_asinh_params,
                          kernel_size=7, padding=3, use_group_norm=True),
            )
        else:
            self.asinh_preprocess = None

    def forward(self, x, asinh_params):
        if self.use_asinh:
            assert asinh_params is not None
        preprocessed_x = self.preprocess3d(x).squeeze(2)

        if self.use_asinh:
            asinh_params_mean = asinh_params.mean(dim=2, keepdim=True)
            asinh_params_var = asinh_params.var(dim=2, keepdim=True)
            normalized_asinh_params = (asinh_params - asinh_params_mean) / torch.sqrt(asinh_params_var + 1e-5)
            preprocessed_asinh_params = self.asinh_preprocess(normalized_asinh_params.unsqueeze(0))
            expanded_asinh_params = preprocessed_asinh_params.expand(preprocessed_x.shape[0], -1, -1, -1).clone()
            output_x = self.backbone(torch.cat((preprocessed_x, expanded_asinh_params), dim=1))
        else:
            output_x = self.backbone(preprocessed_x)

        return output_x
```

(decide to use this new method)

### 06-03

1. try to disable flux error, but this will break the model

2. try to diable psf and CLAHE, and it gives similar results

### 06-04

1. use moving average over per band to process asinh thresholds

2. use moving average over full raw image to process asinh thresholds

3. use moving average over full raw image to process asinh thresholds, and use relu to filter image

```python
processed_raw_images = F.relu(raw_images - self.asinh_thresholds_tensor) * self.asinh_params["scale"]
```

### 06-05

1. use the new method, detach tensors when doing asinh normalization, don't normalize asinh params (forget to add psf params)

2. use the new method, and change asinh scale to 100

3. use the new method, detach tensors when using neural network to preporcess asinh params

4. use moving average over full raw image to process asinh thresholds, and set scale to 100


### 06-06

1. use the new method but trained using refactored code

### 06-12

1. train with fixed thresholds using the latest code

2. train with fixed thresholds using the latest code and fix the `compute_groups` bug in `MetricCollection`

### 06-13

1. use the new method, detach tensors when doing asinh normalization, don't normalize asinh params, exclude psf and clahe, and set initial thresholds to [-0.3856, -0.1059, -0.0336,  0.0073,  0.0569,  0.1658,  0.6423]. (seed is 42) (bug; ignore this exp)

2. repeat 06-13-1 (seed is 218) (bug; ignore this exp)

3. repeat 06-13-1 (seed is 1024) (bug; ignore this exp)

4. repeat 06-13-1 but set asinh_scales_tensor to constant. (seed is 1024) (bug; ignore this exp)

5. use the new method, detach tensors when doing asinh normalization, don't normalize asinh params, exclude psf and clahe, set initial thresholds to [-0.3856, -0.1059, -0.0336,  0.0073,  0.0569,  0.1658,  0.6423], and set asinh_scales_tensor to constant. (only one GPU is used) (seed is 1024)

(use 06-13-5 as the default setting of dynamic asinh)

### 06-14

(hereafter, use one GPU, and set seed to random)

1. run dynamic asinh

2. run dynamic asinh

3. run dynamic asinh

### 06-15

1. run dynamic asinh

2. run dynamic asinh

3. run fixed thresholds asinh

4. run fixed thresholds asinh

5. run fixed thresholds asinh

### 06-16

1. run fixed thresholds asinh

2. run fixed thresholds asinh

3. run moving avg thresholds asinh

4. run moving avg thresholds asinh

5. run moving avg thresholds asinh

(hereafter, use fixed thresholds asinh)

### 06-19

1. run with double detection

2. run with double detection

3. run with double detection

(hereafter, always use double detection)

### 06-24

1. train with the new DC2

2. train with the new DC2

3. train with the new DC2

### 06-25

1. train with new DC2 and old data aug

2. train with new DC2 and new data aug (both transforms) (on Great Lake)

3. train with new DC2 and new data aug (both transforms) (on Great Lake)

4. train with new DC2 and new data aug (both transforms) (on Great Lake)

5. train with new DC2 (no transforms)

### 06-27

1. train using the new data augmentation but rotate/flip plocs in fullcat

2. train using the new data augmentation but only rotate/flip locs

3. train using the old rotate/flip and new random shift

### 06-30

1. old data aug but change rotate and flip func

```python
def plocs_rot90(plocs, image_size):
    plocs_clone = plocs.clone()
    plocs_clone[..., 0], plocs_clone[..., 1] = image_size - plocs[..., 1], plocs[..., 0]
    return plocs_clone


def plocs_vflip(plocs, image_size):
    plocs_clone = plocs.clone()
    plocs_clone[..., 0] = image_size - plocs[..., 0]
    return plocs_clone
```

2. new data aug but use on_mask

```python
 # locations require special logic
if "locs" in datum["tile_catalog"]:
    locs = datum_out["tile_catalog"]["locs"]
    max_sources = locs.shape[-2]
    n_sources = datum_out["tile_catalog"]["n_sources"]
    on_mask = is_on_mask(n_sources=n_sources, max_sources=max_sources)
    for _ in range(rotate_id):
        # Rotate 90 degrees clockwise (in pixel coordinates)
        locs = torch.stack((1 - locs[..., 1], locs[..., 0]), dim=3) * on_mask.unsqueeze(-1)
    if do_flip:
        locs = torch.stack((1 - locs[..., 0], locs[..., 1]), dim=3) * on_mask.unsqueeze(-1)
    datum_out["tile_catalog"]["locs"] = locs
```


3. old data aug

4. old data aug

5. new data aug

6. old data aug but change rotate and flip func

### 07-01

1. new rotate flip + new shift (shift_image=False)

2. new shift (shift_image=False)

3. new rotate flip

4. old rotate flip (bug fixed) + new shift (shift_image=False) + new shift (shift_image=True)

5. old rotate flip (bug fixed) + new shift (shift_image=False) + new shift (shift_image=True) (shift_ub and shift_lb fixed)

6. new shift (shift_image=False) (shift_ub and shift_lb fixed)

7. new rotate flip (rotate_id=1; do_flip=False)

8. without data aug

### 07-06

1. without checkerboard (tiles_to_crop set to 0) (crush due to disk limit)

2. with checkerboard (tiles_to_crop set to 0) (crush due to disk limit)

3. with checkerboard (16-mixed) (tiles_to_crop set to 0)

4. without checkerboard (tiles_to_crop set to 0)

5. with checkerboard (tiles_to_crop set to 0)

(hereafter, always set tiles_to_crop to 0)


### 07-08

1. test MultiDetectEncoder

2. test MultiDetectEncoder
