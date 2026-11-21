<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/pz-prior-unlit-090/result.json>>>
{
  "completed": true,
  "case": "Direct3D-V3 Box face colorization candidate",
  "face": "+Z",
  "faceSlug": "pz",
  "promptId": "1730c2fd-fdc9-48f3-a501-ec23277f286d",
  "promptSHA256": "d2c35ad6e2db480586c80e78051b9041b199809abef6293808e2ebd87a92ad26",
  "images": {
    "generated": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/pz-prior-unlit-090/generated.png",
      "sha256": "bf1334fc3d91f22b42945c440f8a8e8975a97de6a5bff16222eff083e23e64cf",
      "bytes": 763484
    },
    "mask": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/pz-prior-unlit-090/mask.png",
      "sha256": "2f2de10a1afeaf391681a64788adcf615cf96c61576a40e0b6df837b48df61b8",
      "bytes": 47530
    }
  },
  "model": "Qwen Image 2.1 int8",
  "lora": "QI2.1_AnyAngle",
  "loraStrength": 0.0,
  "size": 1024,
  "steps": 10,
  "cfg": 3,
  "seed": 2026113031,
  "sampler": "euler",
  "scheduler": "simple",
  "prompt": "Refine the textures over the COMPLETE already colored 3D model in image 1, using image 2 only as an appearance reference. Preserve image 1's exact camera, projection, full-frame silhouette, body proportions, pose, head angle, facial expression, eye and mouth centers, hairline and hair length, sleeve boundaries, hand and finger locations, dress hem, legs, shoe outlines, heels and all background gaps. Improve every visible material: coherent fine gray-and-white heather knit on the whole dress, blonde hair strands, natural skin and facial detail, and black suede shoes with tan heels and sole edges. Correct rough pixels, obvious texture-transfer defects, missing eye detail and abrupt color patches while keeping every feature at its existing location. Keep the material regions fixed: skin stays skin, fabric stays fabric, hair stays hair, and black shoes stay black. Image 2 must not change image 1's pose, head direction, proportions or framing. Image 3 is matching structural depth, nearer white and background black. This is the +Z front view. Keep the existing downward head angle, expression, eye direction and all feature locations from image 1. Keep all existing overlaps and hidden surfaces hidden. Maintain a soft, even surface-color appearance; do not add dramatic lighting, new strong shadows or highlights. Keep the complete uncropped frame and the same uniform neutral gray background. No rotation, zoom, moved limbs, new objects, text or panels.",
  "promptSource": {
    "kind": "file",
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/pz-prior-unlit.prompt.txt",
    "sha256": "6fd7ad4a4024c74a24e6a8908458b770335e4ecfe33e20fbd30828c35a0c548b"
  },
  "defaults": {
    "profile": "anchored",
    "denoise": 0.9,
    "loraStrength": 0.0,
    "size": 1024,
    "steps": 10,
    "cfg": 3,
    "seed": 2026113031,
    "sampler": "euler",
    "scheduler": "simple"
  },
  "inputOrder": [
    "target full-view RGB",
    "original reference photograph",
    "target depth"
  ],
  "server": "http://127.0.0.1:8198",
  "runnerSha256": "4cf769b1d3ecc695d11324d32637f1df1deeaae251046f314e94d9aa93898698",
  "profile": "anchored",
  "denoise": 0.9,
  "initialLatent": "VAE-encoded target RGB",
  "serverExecutionSeconds": 63.465,
  "clientWaitSeconds": 64.10875401899102,
  "collectionOnly": false,
  "cachedNodes": [
    "6",
    "7",
    "8",
    "15",
    "19",
    "9"
  ],
  "inputsSHA256": {
    "target-color.png": "e2462d3bf25d2f17ad533966360b08a7d6ddc298b0ef9b8a864e71c6dc384f65",
    "target-depth.png": "01bcc73ed12b1a5501398bccbd4b6049f8b31ba0587ceb9093ae56615411ef08",
    "original.png": "9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b"
  },
  "vram": {
    "measurement": "nvidia-smi sampled whole-device used VRAM",
    "intervalMilliseconds": 200,
    "sampleCount": 315,
    "peakSampledUsedMiB": 13732.0,
    "limitation": "Includes driver/other processes; not allocator peak; may miss short peaks."
  },
  "boxModified": false,
  "nextStage": "Review this full-view candidate, register generated RGB/mask against its guide, and compose only selected face RGB while preserving Direct3D structural alpha and geometry."
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/px-prior-unlit-090/result.json>>>
{
  "completed": true,
  "case": "Direct3D-V3 Box face colorization candidate",
  "face": "+X",
  "faceSlug": "px",
  "promptId": "e598115d-b2fe-40f5-8c38-5ab2161503bf",
  "promptSHA256": "1eea579079fc92722946fd074a3c84cb0566cdb8f9a896c224bf13a7dc01f53e",
  "images": {
    "generated": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/px-prior-unlit-090/generated.png",
      "sha256": "fe4369af8a776c21adc3187515b89f8c30f3893ff13d39b10536acbd914a0366",
      "bytes": 752877
    },
    "mask": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/px-prior-unlit-090/mask.png",
      "sha256": "9ebb9c8099c2cbcb2da70ecb9120d805c234caf78127e17e7bbc3c3b716a4741",
      "bytes": 46007
    }
  },
  "model": "Qwen Image 2.1 int8",
  "lora": "QI2.1_AnyAngle",
  "loraStrength": 0.0,
  "size": 1024,
  "steps": 10,
  "cfg": 3,
  "seed": 2026113031,
  "sampler": "euler",
  "scheduler": "simple",
  "prompt": "Refine the textures over the COMPLETE already colored 3D model in image 1, using image 2 only as an appearance reference. Preserve image 1's exact camera, projection, full-frame silhouette, body proportions, pose, head angle, facial expression, eye and mouth centers, hairline and hair length, sleeve boundaries, hand and finger locations, dress hem, legs, shoe outlines, heels and all background gaps. Improve every visible material: coherent fine gray-and-white heather knit on the whole dress, blonde hair strands, natural skin and facial detail, and black suede shoes with tan heels and sole edges. Correct rough pixels, obvious texture-transfer defects, missing eye detail and abrupt color patches while keeping every feature at its existing location. Keep the material regions fixed: skin stays skin, fabric stays fabric, hair stays hair, and black shoes stay black. Image 2 must not change image 1's pose, head direction, proportions or framing. Image 3 is matching structural depth, nearer white and background black. This is the +X side view looking toward -X. Keep this exact side camera, head profile and rearward visible surfaces. Do not turn the head toward the camera or reveal hidden facial features. Keep all existing overlaps and hidden surfaces hidden. Maintain a soft, even surface-color appearance; do not add dramatic lighting, new strong shadows or highlights. Keep the complete uncropped frame and the same uniform neutral gray background. No rotation, zoom, moved limbs, new objects, text or panels.",
  "promptSource": {
    "kind": "file",
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/px-prior-unlit.prompt.txt",
    "sha256": "be19d89295b5b8e2a6b6658c5f3991f15b14834cf51eabc5a96ace48ae0e38c6"
  },
  "defaults": {
    "profile": "anchored",
    "denoise": 0.9,
    "loraStrength": 0.0,
    "size": 1024,
    "steps": 10,
    "cfg": 3,
    "seed": 2026113031,
    "sampler": "euler",
    "scheduler": "simple"
  },
  "inputOrder": [
    "target full-view RGB",
    "original reference photograph",
    "target depth"
  ],
  "server": "http://127.0.0.1:8198",
  "runnerSha256": "4cf769b1d3ecc695d11324d32637f1df1deeaae251046f314e94d9aa93898698",
  "profile": "anchored",
  "denoise": 0.9,
  "initialLatent": "VAE-encoded target RGB",
  "serverExecutionSeconds": 63.021,
  "clientWaitSeconds": 64.20000033799442,
  "collectionOnly": false,
  "cachedNodes": [
    "6",
    "7",
    "8",
    "15",
    "19",
    "9"
  ],
  "inputsSHA256": {
    "target-color.png": "724007e9d9ca1f10410b00ea9da4c35344f850cfcb5d8d69f1ca0d4f72cc6c36",
    "target-depth.png": "e009e2d877c6b68b0b935d3fc5977f94e5db5946bfc1d952ee596ce260a0f9eb",
    "original.png": "9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b"
  },
  "vram": {
    "measurement": "nvidia-smi sampled whole-device used VRAM",
    "intervalMilliseconds": 200,
    "sampleCount": 314,
    "peakSampledUsedMiB": 13721.0,
    "limitation": "Includes driver/other processes; not allocator peak; may miss short peaks."
  },
  "boxModified": false,
  "nextStage": "Review this full-view candidate, register generated RGB/mask against its guide, and compose only selected face RGB while preserving Direct3D structural alpha and geometry."
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nx-prior-unlit-090/result.json>>>
{
  "completed": true,
  "case": "Direct3D-V3 Box face colorization candidate",
  "face": "-X",
  "faceSlug": "nx",
  "promptId": "119672f8-947a-4d9c-ae84-f37078200abb",
  "promptSHA256": "d08dd5e70c4f9c2daa3306fc16bdf4c4c9c7922319a0b06c9dab934b4ee8fea8",
  "images": {
    "generated": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nx-prior-unlit-090/generated.png",
      "sha256": "08b8d01f9e8738a7dab9741b8d3986ccc578bbd0ddc8c6971d99d3cde946d809",
      "bytes": 904646
    },
    "mask": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nx-prior-unlit-090/mask.png",
      "sha256": "9ab6572ceb599ae4da7725d3612d10235cf250e66befbb9a398a1027bde1145b",
      "bytes": 45029
    }
  },
  "model": "Qwen Image 2.1 int8",
  "lora": "QI2.1_AnyAngle",
  "loraStrength": 0.0,
  "size": 1024,
  "steps": 10,
  "cfg": 3,
  "seed": 2026113031,
  "sampler": "euler",
  "scheduler": "simple",
  "prompt": "Refine the textures over the COMPLETE already colored 3D model in image 1, using image 2 only as an appearance reference. Preserve image 1's exact camera, projection, full-frame silhouette, body proportions, pose, head angle, facial expression, eye and mouth centers, hairline and hair length, sleeve boundaries, hand and finger locations, dress hem, legs, shoe outlines, heels and all background gaps. Improve every visible material: coherent fine gray-and-white heather knit on the whole dress, blonde hair strands, natural skin and facial detail, and black suede shoes with tan heels and sole edges. Correct rough pixels, obvious texture-transfer defects, missing eye detail and abrupt color patches while keeping every feature at its existing location. Keep the material regions fixed: skin stays skin, fabric stays fabric, hair stays hair, and black shoes stay black. Image 2 must not change image 1's pose, head direction, proportions or framing. Image 3 is matching structural depth, nearer white and background black. This is the -X side view looking toward +X. Keep this exact side camera, head profile, limb overlap and all occlusions. Do not turn the head toward the camera or reveal hidden facial features. Keep all existing overlaps and hidden surfaces hidden. Maintain a soft, even surface-color appearance; do not add dramatic lighting, new strong shadows or highlights. Keep the complete uncropped frame and the same uniform neutral gray background. No rotation, zoom, moved limbs, new objects, text or panels.",
  "promptSource": {
    "kind": "file",
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nx-prior-unlit.prompt.txt",
    "sha256": "5ce15c93b3f35f676269e0d8903a7f02fe0a95651d08daf6a9b56c7254d44b61"
  },
  "defaults": {
    "profile": "anchored",
    "denoise": 0.9,
    "loraStrength": 0.0,
    "size": 1024,
    "steps": 10,
    "cfg": 3,
    "seed": 2026113031,
    "sampler": "euler",
    "scheduler": "simple"
  },
  "inputOrder": [
    "target full-view RGB",
    "original reference photograph",
    "target depth"
  ],
  "server": "http://127.0.0.1:8198",
  "runnerSha256": "4cf769b1d3ecc695d11324d32637f1df1deeaae251046f314e94d9aa93898698",
  "profile": "anchored",
  "denoise": 0.9,
  "initialLatent": "VAE-encoded target RGB",
  "serverExecutionSeconds": 63.852,
  "clientWaitSeconds": 64.11461554199923,
  "collectionOnly": false,
  "cachedNodes": [
    "6",
    "7",
    "8",
    "15",
    "19",
    "9"
  ],
  "inputsSHA256": {
    "target-color.png": "4bd107465e68ec61a85e96b45fe1cdbdb914fa66674e2004c4a66335bb06a6e1",
    "target-depth.png": "310afe59da126aade6fda51d2bf13d5063b4a0173b7528c2a98610519d648aa1",
    "original.png": "9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b"
  },
  "vram": {
    "measurement": "nvidia-smi sampled whole-device used VRAM",
    "intervalMilliseconds": 200,
    "sampleCount": 312,
    "peakSampledUsedMiB": 13720.0,
    "limitation": "Includes driver/other processes; not allocator peak; may miss short peaks."
  },
  "boxModified": false,
  "nextStage": "Review this full-view candidate, register generated RGB/mask against its guide, and compose only selected face RGB while preserving Direct3D structural alpha and geometry."
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nz-prior-unlit-090/result.json>>>
{
  "completed": true,
  "case": "Direct3D-V3 Box face colorization candidate",
  "face": "-Z",
  "faceSlug": "nz",
  "promptId": "6607f121-f6e9-4287-979f-3ceb70c22130",
  "promptSHA256": "a408a5a6fd2c2f09972702fcd213e58a99f20fe669417434f7a7012495b949ba",
  "images": {
    "generated": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nz-prior-unlit-090/generated.png",
      "sha256": "becd99139e759715e3fbc71ab2df81caa626c4509c535a57ec5092d302734c3b",
      "bytes": 871359
    },
    "mask": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nz-prior-unlit-090/mask.png",
      "sha256": "1fcf982e1f33189fcbbd9ed505c2404b6830f32c966655358b576d7fe40f324e",
      "bytes": 47103
    }
  },
  "model": "Qwen Image 2.1 int8",
  "lora": "QI2.1_AnyAngle",
  "loraStrength": 0.0,
  "size": 1024,
  "steps": 10,
  "cfg": 3,
  "seed": 2026113031,
  "sampler": "euler",
  "scheduler": "simple",
  "prompt": "Refine the textures over the COMPLETE already colored 3D model in image 1, using image 2 only as an appearance reference. Preserve image 1's exact camera, projection, full-frame silhouette, body proportions, pose, head angle, facial expression, eye and mouth centers, hairline and hair length, sleeve boundaries, hand and finger locations, dress hem, legs, shoe outlines, heels and all background gaps. Improve every visible material: coherent fine gray-and-white heather knit on the whole dress, blonde hair strands, natural skin and facial detail, and black suede shoes with tan heels and sole edges. Correct rough pixels, obvious texture-transfer defects, missing eye detail and abrupt color patches while keeping every feature at its existing location. Keep the material regions fixed: skin stays skin, fabric stays fabric, hair stays hair, and black shoes stay black. Image 2 must not change image 1's pose, head direction, proportions or framing. Image 3 is matching structural depth, nearer white and background black. This is the -Z BACK view. Show the back of the hair and the rear of the dress. Preserve the back-facing head and body. Do not paint a face, eyes, nose or mouth on the back of the head or turn the person toward the camera. Keep all existing overlaps and hidden surfaces hidden. Maintain a soft, even surface-color appearance; do not add dramatic lighting, new strong shadows or highlights. Keep the complete uncropped frame and the same uniform neutral gray background. No rotation, zoom, moved limbs, new objects, text or panels.",
  "promptSource": {
    "kind": "file",
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/nz-prior-unlit.prompt.txt",
    "sha256": "255f48212fbb4aa8ce65943d753f13e316b64afb1ed6ff6189e25d46fb15621f"
  },
  "defaults": {
    "profile": "anchored",
    "denoise": 0.9,
    "loraStrength": 0.0,
    "size": 1024,
    "steps": 10,
    "cfg": 3,
    "seed": 2026113031,
    "sampler": "euler",
    "scheduler": "simple"
  },
  "inputOrder": [
    "target full-view RGB",
    "original reference photograph",
    "target depth"
  ],
  "server": "http://127.0.0.1:8198",
  "runnerSha256": "4cf769b1d3ecc695d11324d32637f1df1deeaae251046f314e94d9aa93898698",
  "profile": "anchored",
  "denoise": 0.9,
  "initialLatent": "VAE-encoded target RGB",
  "serverExecutionSeconds": 63.081,
  "clientWaitSeconds": 64.11017213098239,
  "collectionOnly": false,
  "cachedNodes": [
    "6",
    "7",
    "8",
    "15",
    "19",
    "9"
  ],
  "inputsSHA256": {
    "target-color.png": "7bedc96579c907c168ce9e22a5dbac23e9dfe04d710fa96790b705132b327ec7",
    "target-depth.png": "08d4a37a4ec5acfea03cf11b4c64105a2bfeeb9c4f1b99bb319a5cc589383d76",
    "original.png": "9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b"
  },
  "vram": {
    "measurement": "nvidia-smi sampled whole-device used VRAM",
    "intervalMilliseconds": 200,
    "sampleCount": 313,
    "peakSampledUsedMiB": 13720.0,
    "limitation": "Includes driver/other processes; not allocator peak; may miss short peaks."
  },
  "boxModified": false,
  "nextStage": "Review this full-view candidate, register generated RGB/mask against its guide, and compose only selected face RGB while preserving Direct3D structural alpha and geometry."
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/py-prior-unlit-090/result.json>>>
{
  "completed": true,
  "case": "Direct3D-V3 Box face colorization candidate",
  "face": "+Y",
  "faceSlug": "py",
  "promptId": "88f0041e-4c5e-485e-95dc-0220453b6f7e",
  "promptSHA256": "39bb47cbe210166233dcd2e7195f7450881042d6246e09980285125ee585f16a",
  "images": {
    "generated": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/py-prior-unlit-090/generated.png",
      "sha256": "e4d9d033add8ef7f0ab6ec4e2c89bc713ad766ce293dc25abf1da9159f15bb27",
      "bytes": 993537
    },
    "mask": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/py-prior-unlit-090/mask.png",
      "sha256": "de39d421c5a0cd3a76eae92a1ef3ab3fcd31df98a6be8ca74e2e3e7f06ec2597",
      "bytes": 52622
    }
  },
  "model": "Qwen Image 2.1 int8",
  "lora": "QI2.1_AnyAngle",
  "loraStrength": 0.0,
  "size": 1024,
  "steps": 10,
  "cfg": 3,
  "seed": 2026113031,
  "sampler": "euler",
  "scheduler": "simple",
  "prompt": "Refine the textures over the COMPLETE already colored 3D model in image 1, using image 2 only as an appearance reference. Preserve image 1's exact camera, projection, full-frame silhouette, body proportions, pose, head angle, facial expression, eye and mouth centers, hairline and hair length, sleeve boundaries, hand and finger locations, dress hem, legs, shoe outlines, heels and all background gaps. Improve every visible material: coherent fine gray-and-white heather knit on the whole dress, blonde hair strands, natural skin and facial detail, and black suede shoes with tan heels and sole edges. Correct rough pixels, obvious texture-transfer defects, missing eye detail and abrupt color patches while keeping every feature at its existing location. Keep the material regions fixed: skin stays skin, fabric stays fabric, hair stays hair, and black shoes stay black. Image 2 must not change image 1's pose, head direction, proportions or framing. Image 3 is matching structural depth, nearer white and background black. This is an orthographic TOP-DOWN view from +Y looking downward. The head and shoulders occlude the body; preserve this exact unusual overhead projection and every overlap. Only color surfaces already visible in image 1. Never rearrange this into an upright full-body or front view, reveal hidden limbs, or add a face. Keep all existing overlaps and hidden surfaces hidden. Maintain a soft, even surface-color appearance; do not add dramatic lighting, new strong shadows or highlights. Keep the complete uncropped frame and the same uniform neutral gray background. No rotation, zoom, moved limbs, new objects, text or panels.",
  "promptSource": {
    "kind": "file",
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/py-prior-unlit.prompt.txt",
    "sha256": "5ca9d3f32736555704edda7b6958993e2fc7bcef294a5194fddb15a7be7af311"
  },
  "defaults": {
    "profile": "anchored",
    "denoise": 0.9,
    "loraStrength": 0.0,
    "size": 1024,
    "steps": 10,
    "cfg": 3,
    "seed": 2026113031,
    "sampler": "euler",
    "scheduler": "simple"
  },
  "inputOrder": [
    "target full-view RGB",
    "original reference photograph",
    "target depth"
  ],
  "server": "http://127.0.0.1:8198",
  "runnerSha256": "4cf769b1d3ecc695d11324d32637f1df1deeaae251046f314e94d9aa93898698",
  "profile": "anchored",
  "denoise": 0.9,
  "initialLatent": "VAE-encoded target RGB",
  "serverExecutionSeconds": 62.775,
  "clientWaitSeconds": 64.10782870399999,
  "collectionOnly": false,
  "cachedNodes": [
    "6",
    "7",
    "8",
    "15",
    "19",
    "9"
  ],
  "inputsSHA256": {
    "target-color.png": "b4b1c28a952cc1dcbab8b9116e641afb5a9298848ec877f7bc5b6a9fc2865a78",
    "target-depth.png": "7d50184d46827611826855cee805c331f72a63c095e9bca8e8de92866b078c70",
    "original.png": "9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b"
  },
  "vram": {
    "measurement": "nvidia-smi sampled whole-device used VRAM",
    "intervalMilliseconds": 200,
    "sampleCount": 313,
    "peakSampledUsedMiB": 13720.0,
    "limitation": "Includes driver/other processes; not allocator peak; may miss short peaks."
  },
  "boxModified": false,
  "nextStage": "Review this full-view candidate, register generated RGB/mask against its guide, and compose only selected face RGB while preserving Direct3D structural alpha and geometry."
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/ny-prior-unlit-090/result.json>>>
{
  "completed": true,
  "case": "Direct3D-V3 Box face colorization candidate",
  "face": "-Y",
  "faceSlug": "ny",
  "promptId": "fcb1763c-a477-45aa-808e-8826badc9b09",
  "promptSHA256": "d316c3c49671221ed7e18d93ead8c913985e947123b39c20ae6dcb901757a4df",
  "images": {
    "generated": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/ny-prior-unlit-090/generated.png",
      "sha256": "5bd5adecf6107c815d125679cc0390e17c0a6f165df72d78bfe18537c7150438",
      "bytes": 966473
    },
    "mask": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/ny-prior-unlit-090/mask.png",
      "sha256": "337f010ffc24116e4ff3d2eef3df83a0fd36241dd5de6b8a1cdc482fd2b66f68",
      "bytes": 51989
    }
  },
  "model": "Qwen Image 2.1 int8",
  "lora": "QI2.1_AnyAngle",
  "loraStrength": 0.0,
  "size": 1024,
  "steps": 10,
  "cfg": 3,
  "seed": 2026113031,
  "sampler": "euler",
  "scheduler": "simple",
  "prompt": "Refine the textures over the COMPLETE already colored 3D model in image 1, using image 2 only as an appearance reference. Preserve image 1's exact camera, projection, full-frame silhouette, body proportions, pose, head angle, facial expression, eye and mouth centers, hairline and hair length, sleeve boundaries, hand and finger locations, dress hem, legs, shoe outlines, heels and all background gaps. Improve every visible material: coherent fine gray-and-white heather knit on the whole dress, blonde hair strands, natural skin and facial detail, and black suede shoes with tan heels and sole edges. Correct rough pixels, obvious texture-transfer defects, missing eye detail and abrupt color patches while keeping every feature at its existing location. Keep the material regions fixed: skin stays skin, fabric stays fabric, hair stays hair, and black shoes stay black. Image 2 must not change image 1's pose, head direction, proportions or framing. Image 3 is matching structural depth, nearer white and background black. This is an orthographic UNDERSIDE view from -Y looking upward. The shoes and their soles are nearest to the camera and occlude the body; preserve this exact unusual underside projection and every overlap. Color the visible shoe soles tan and black. Only color surfaces already visible in image 1. Never rearrange this into an upright full-body or front view, reveal hidden limbs, or add a face. Keep all existing overlaps and hidden surfaces hidden. Maintain a soft, even surface-color appearance; do not add dramatic lighting, new strong shadows or highlights. Keep the complete uncropped frame and the same uniform neutral gray background. No rotation, zoom, moved limbs, new objects, text or panels.",
  "promptSource": {
    "kind": "file",
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/qwen/ny-prior-unlit.prompt.txt",
    "sha256": "b207430da112e774156f3cf189b1bdfffc166edbe74d1bcc003574b5d2bda14f"
  },
  "defaults": {
    "profile": "anchored",
    "denoise": 0.9,
    "loraStrength": 0.0,
    "size": 1024,
    "steps": 10,
    "cfg": 3,
    "seed": 2026113031,
    "sampler": "euler",
    "scheduler": "simple"
  },
  "inputOrder": [
    "target full-view RGB",
    "original reference photograph",
    "target depth"
  ],
  "server": "http://127.0.0.1:8198",
  "runnerSha256": "4cf769b1d3ecc695d11324d32637f1df1deeaae251046f314e94d9aa93898698",
  "profile": "anchored",
  "denoise": 0.9,
  "initialLatent": "VAE-encoded target RGB",
  "serverExecutionSeconds": 63.556,
  "clientWaitSeconds": 64.11483015998965,
  "collectionOnly": false,
  "cachedNodes": [
    "6",
    "7",
    "8",
    "15",
    "19",
    "9"
  ],
  "inputsSHA256": {
    "target-color.png": "6b1736a740e7f35caa21018aaed3ee189728df6aedd75e1df0ac07a42758ab1a",
    "target-depth.png": "c07f869e055716a6787c9026e96c1af3b26be469359486605f83c7b2aeef1a9c",
    "original.png": "9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b"
  },
  "vram": {
    "measurement": "nvidia-smi sampled whole-device used VRAM",
    "intervalMilliseconds": 200,
    "sampleCount": 312,
    "peakSampledUsedMiB": 13720.0,
    "limitation": "Includes driver/other processes; not allocator peak; may miss short peaks."
  },
  "boxModified": false,
  "nextStage": "Review this full-view candidate, register generated RGB/mask against its guide, and compose only selected face RGB while preserving Direct3D structural alpha and geometry."
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/prepare-verification.json>>>
{
  "completed": true,
  "source": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-v3-20261121/rgb1024/canonical.glb",
    "bytes": 27002988,
    "sha256": "d7f5e2eb289a4565815782165486485cb9a161094695c1786141c81aea6064ee"
  },
  "output": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/canonical.smooth-normals.glb",
    "bytes": 36003860,
    "sha256": "7d71a13f73b15d99a030b453a2c6c8954c9c17646cc47b406bc6048711e5171d"
  },
  "identityMeta": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/identity-canonical.json",
    "bytes": 595,
    "sha256": "eaa5927f2cce68e5b59d8b522a25fbcaa03cb1be829dbee7162d6e8d73db794e"
  },
  "helper": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/prepare_smooth_normals.py",
    "bytes": 15846,
    "sha256": "375a6d1ae59aa5e86ff990c6b85db4b38ceb99a119ff7ed45f34eceda1f978ab"
  },
  "inspectorScript": {
    "path": "/home/dev/Projects/box-impostor-lab/third_party/3D-Inspector-skills/skills/3d-inspector/scripts/build_viewer.py",
    "bytes": 37806,
    "sha256": "ed0d642c76e73b4acdec5a890d3ed82612949ea977b72ec85cfc76b623e3130d"
  },
  "trimeshVersion": "5.2.1",
  "normalMethod": "Official Inspector: trimesh.load(force='scene', process=False).geometry[only].vertex_normals; corner-angle weighting; identity normal matrix; float64 re-unitize; float32 little-endian",
  "positionsAndIndices": {
    "POSITION": {
      "rawBytesEqual": true,
      "sha256": "8715015e2d5373d7f4d20b1dc2e698658bb0a3ca0a49ef43a9fe77bcb43a9147",
      "count": 750060,
      "dtype": "float32"
    },
    "indices": {
      "rawBytesEqual": true,
      "sha256": "c544ea7d727fe298476a3ca381f7a0ae4f42d9ff00595878973e3df04663dd6b",
      "count": 4500372,
      "dtype": "uint32"
    }
  },
  "sceneJson": {
    "scene": {
      "rawJsonBytesEqual": true,
      "sha256": "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9"
    },
    "scenes": {
      "rawJsonBytesEqual": true,
      "sha256": "09a90a0a479492a2a12bd40b28da366ef440490071c092a6b6f971267867a454"
    },
    "nodes": {
      "rawJsonBytesEqual": true,
      "sha256": "03c76ac2d01cadd2d4c1687e3bb232b9b5d44ef8c89e924b0de766582d920d45"
    }
  },
  "originalBinPrefix": {
    "bytes": 27002208,
    "sha256": "3a57656b7ed8027916ef212fb804392b5b3630494ca2c0973d73f9e89e8474b0",
    "equal": true
  },
  "originalDocumentUnchangedExceptNormalAdditionAndBufferLength": true,
  "vertices": 750060,
  "triangles": 1500124,
  "normalData": {
    "sha256": "d81eeb6d139f59e186d8ff836dc47cf31a945b9378638836c0a148e3a17c6c2b",
    "bytes": 9000720,
    "zeroLengthCount": 0,
    "minLength": 0.9999999514809682,
    "maxLength": 1.0000000502189228
  },
  "elapsedSeconds": 1.8863960360176861
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/native-comparison.json>>>
{
  "before": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/native512/manifest.json",
    "bytes": 12543,
    "sha256": "f7c86b4591cddea531560e03e749cceb9a1173808ddb49c6a12775b1bf0a4445"
  },
  "after": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/native512/manifest.json",
    "bytes": 12543,
    "sha256": "e6cc1ecbc28ff93ef212d1225af1f06be84cfa29478a07e2843c7050ef7802ee"
  },
  "metadata": {
    "schemaVersion": true,
    "mode": true,
    "layout": true,
    "resolution": true,
    "tightBounds": true,
    "bakeBox": true,
    "transform": true,
    "inverseTransform": true,
    "faceOrder": true,
    "coordinateSystem": true,
    "backend": true,
    "versions": true
  },
  "textures": {
    "colorMap": {
      "beforeSha256": "9815ccedec2b4183a9dcc2da554bc42edb7e21b35f4ae5b0aaa5fb85c794068b",
      "afterSha256": "9815ccedec2b4183a9dcc2da554bc42edb7e21b35f4ae5b0aaa5fb85c794068b",
      "rawBytesEqual": true,
      "differentUint16Values": 0,
      "alphaUint16Equal": true,
      "differentAlphaUint16Values": 0
    },
    "normalMap": {
      "beforeSha256": "bbe3d460adb42932e843d4579e3dd2306bb17712ba130eda2cceaa445fcb2b38",
      "afterSha256": "bfe952bb631c95666f8054a2940cf28a0f3c721680549ee4f0d53987e0ed7741",
      "rawBytesEqual": false,
      "differentUint16Values": 1562754,
      "alphaUint16Equal": true,
      "differentAlphaUint16Values": 0,
      "differentRgbPixels": 552462,
      "maxAbsRgbDifference": 0.4012451171875,
      "foregroundPixels": 553272
    },
    "intervalMap": {
      "beforeSha256": "9458e5cda155b2d4e5b5482b229484abf0194f3d7e7fed8e40329f47f4dc2fd0",
      "afterSha256": "9458e5cda155b2d4e5b5482b229484abf0194f3d7e7fed8e40329f47f4dc2fd0",
      "rawBytesEqual": true,
      "differentUint16Values": 0,
      "alphaUint16Equal": true,
      "differentAlphaUint16Values": 0
    },
    "rangeMap": {
      "beforeSha256": "f638fb30287fdffe24c4802d6fd2dd432b615918b00235d81fa577bfb5eeabb7",
      "afterSha256": "f638fb30287fdffe24c4802d6fd2dd432b615918b00235d81fa577bfb5eeabb7",
      "rawBytesEqual": true,
      "differentUint16Values": 0,
      "alphaUint16Equal": true,
      "differentAlphaUint16Values": 0
    }
  },
  "faces": {
    "+X": {
      "cameraAndFaceMetadataEqual": true,
      "depthBytesEqual": true,
      "beforeDepthSha256": "fd7d681090c1adb203f9eae512b7bf4b1ec8f7271edbaa91e35943937877a013",
      "afterDepthSha256": "fd7d681090c1adb203f9eae512b7bf4b1ec8f7271edbaa91e35943937877a013",
      "guideAlphaEqual": true,
      "guideRgbaEqual": true
    },
    "-X": {
      "cameraAndFaceMetadataEqual": true,
      "depthBytesEqual": true,
      "beforeDepthSha256": "59a8c6d4bcfc25ea1d4ded109b7fb8cc9634618b7a77a0903a0d46809607b8d2",
      "afterDepthSha256": "59a8c6d4bcfc25ea1d4ded109b7fb8cc9634618b7a77a0903a0d46809607b8d2",
      "guideAlphaEqual": true,
      "guideRgbaEqual": true
    },
    "+Y": {
      "cameraAndFaceMetadataEqual": true,
      "depthBytesEqual": true,
      "beforeDepthSha256": "bf96bf91e88e5eb9eca4c426eec95d2bf311a616742b367f4abfc16e80ca149f",
      "afterDepthSha256": "bf96bf91e88e5eb9eca4c426eec95d2bf311a616742b367f4abfc16e80ca149f",
      "guideAlphaEqual": true,
      "guideRgbaEqual": true
    },
    "-Y": {
      "cameraAndFaceMetadataEqual": true,
      "depthBytesEqual": true,
      "beforeDepthSha256": "c6c32bc26f4f0a119ad7c20507725e26fc80590c73b7d1721235f5ebc1592f95",
      "afterDepthSha256": "c6c32bc26f4f0a119ad7c20507725e26fc80590c73b7d1721235f5ebc1592f95",
      "guideAlphaEqual": true,
      "guideRgbaEqual": true
    },
    "+Z": {
      "cameraAndFaceMetadataEqual": true,
      "depthBytesEqual": true,
      "beforeDepthSha256": "e92e21c5745e87b319478b79cee3ddc3a0970840300f12d659740a94643ce87d",
      "afterDepthSha256": "e92e21c5745e87b319478b79cee3ddc3a0970840300f12d659740a94643ce87d",
      "guideAlphaEqual": true,
      "guideRgbaEqual": true
    },
    "-Z": {
      "cameraAndFaceMetadataEqual": true,
      "depthBytesEqual": true,
      "beforeDepthSha256": "0d824bdc68140297532d555c152a4a132b4e925c7470fd849cca667fb84cf9d8",
      "afterDepthSha256": "0d824bdc68140297532d555c152a4a132b4e925c7470fd849cca667fb84cf9d8",
      "guideAlphaEqual": true,
      "guideRgbaEqual": true
    }
  },
  "allStructuralChecksPassed": true,
  "normalRgbChanged": true,
  "renderAdapterBefore": {
    "renderer": "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)"
  },
  "renderAdapterAfter": {
    "renderer": "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)"
  },
  "scope": "Separate bake candidate only; existing native512, prepared guides and Qwen assets are unchanged."
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/normal-revision-result.json>>>
{
  "completed": true,
  "method": "direct3d-normal-only-revision-v1",
  "helper": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/apply_normal_revision.py",
    "sha256": "46e0a8f71966f91b943a3db03925eef8007e0b8634d5ac861794790d8e92dbdc",
    "bytes": 16609
  },
  "dependencies": [
    {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/prepare_mesh_box.py",
      "sha256": "1a11567411ca0c7502d98125c7580a31cd985867a65baedad7f7cf19b5c2af90",
      "bytes": 17541
    },
    {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/compose_faces.py",
      "sha256": "ca10a7373535664c3dd8b3a9289351a34e15865e2db0200f66e89a460d4faa1a",
      "bytes": 28336
    }
  ],
  "baselines": {
    "gray": {
      "sourceManifest": {
        "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/prepared/baseline512/manifest.json",
        "sha256": "0fcd3e3d57f75aa35f8ec07af8541dca0c0dfe56167b49fef65f373533fd88ba",
        "bytes": 13073
      },
      "manifest": {
        "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/baselines/gray/manifest.json",
        "sha256": "471d25a9b8f504368f12ac459d367eadf51e2bb2101b76ec886e997e73660e50",
        "bytes": 13205
      },
      "qa": {
        "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/baselines/gray/qa/normal-only-3fcb2f986f158c05.json",
        "sha256": "aa3092c0dfe32ae104953abb01ebfffee719aaece1254adfa017ff3929cd9719",
        "bytes": 5881
      },
      "normalAlphaUint16DifferenceCount": 0,
      "normalRgbChangedPixels": 552462,
      "allOtherOriginalFilesIdentical": true,
      "originalSourceFilesUnchanged": true
    },
    "prior": {
      "sourceManifest": {
        "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/color-prior/baseline512/manifest.json",
        "sha256": "e6e4dc0039029141fddc6655c9670a20dd6955b79bfb21bc114d2db601c13e50",
        "bytes": 13212
      },
      "manifest": {
        "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/baselines/prior/manifest.json",
        "sha256": "556b555829562c9cec35f97aa59e438eefdbd36f6619c4f1043b985c86568c87",
        "bytes": 13205
      },
      "qa": {
        "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/baselines/prior/qa/normal-only-badb51598fbefcdc.json",
        "sha256": "8d7af72eb0532221739c4e319c54647d3e39f0d0118f1162179bec972305d62a",
        "bytes": 6108
      },
      "normalAlphaUint16DifferenceCount": 0,
      "normalRgbChangedPixels": 552462,
      "allOtherOriginalFilesIdentical": true,
      "originalSourceFilesUnchanged": true
    }
  },
  "guides": {
    "result": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/guides-unlit/result.json",
      "sha256": "84b9437c8c110cb91cf56dcdacadd0d7539eaf7e99116045aa4dfb76ae878c31",
      "bytes": 19513
    },
    "guides": [
      {
        "face": "+X",
        "slug": "px",
        "sourceGuide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/color-prior/guides-unlit/px/guide.json",
          "sha256": "75bac901d3ad3b6425c2cceea8043cc566e28c9181c5e5b9451742f1e473cd4a",
          "bytes": 13016
        },
        "guide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/guides-unlit/px/guide.json",
          "sha256": "fa81bde6988552637de87e64a1d62c107d76b398c8f0599985eca1ac7d56e8bd",
          "bytes": 16350
        },
        "allImageRecordsUnchanged": true,
        "cameraMappingProxyUnchanged": true,
        "unlitReplay": {
          "normalMapReadByReplay": false,
          "allSevenFilesByteIdentical": true,
          "files": {
            "rgb": {
              "sha256": "724007e9d9ca1f10410b00ea9da4c35344f850cfcb5d8d69f1ca0d4f72cc6c36",
              "bytes": 186854,
              "allBytesEqual": true
            },
            "albedo": {
              "sha256": "724007e9d9ca1f10410b00ea9da4c35344f850cfcb5d8d69f1ca0d4f72cc6c36",
              "bytes": 186854,
              "allBytesEqual": true
            },
            "alpha": {
              "sha256": "c3e0317cbe7e251ae32d56c7f58263ff6baa77c7744fa170869cea59d188a69d",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "alphaPng": {
              "sha256": "dd250ddb91d60fc61694cdb8ae32da906e00ff33c3071b56728093a40e806e17",
              "bytes": 14197,
              "allBytesEqual": true
            },
            "depth": {
              "sha256": "46608bddd32bc5b065ed16653ed90fdcf7300328215df288b95bc0f3ac548c50",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "depthPreview": {
              "sha256": "4357108a4bd409638705f384a830980ae074f415395ec6a782f5433efd6308d3",
              "bytes": 55708,
              "allBytesEqual": true
            },
            "depth512": {
              "sha256": "e009e2d877c6b68b0b935d3fc5977f94e5db5946bfc1d952ee596ce260a0f9eb",
              "bytes": 20269,
              "allBytesEqual": true
            }
          },
          "dependencies": [
            "colorMap RGB and alpha",
            "native depth",
            "fixed camera/affine",
            "sRGB conversion and fixed background"
          ],
          "method": "In-memory replay of unlit albedo/alpha/depth PNG and NPY encodings; no normal or lighting input"
        }
      },
      {
        "face": "-X",
        "slug": "nx",
        "sourceGuide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/color-prior/guides-unlit/nx/guide.json",
          "sha256": "dd477148417c13a82e36b0f25437457a4ea5f2cb5387f44325e0e96e3adf1002",
          "bytes": 13002
        },
        "guide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/guides-unlit/nx/guide.json",
          "sha256": "f0eac733c7f7c6191217a15ea29c4bbd092f6c7ab221dee46742bffeea7ea93c",
          "bytes": 16336
        },
        "allImageRecordsUnchanged": true,
        "cameraMappingProxyUnchanged": true,
        "unlitReplay": {
          "normalMapReadByReplay": false,
          "allSevenFilesByteIdentical": true,
          "files": {
            "rgb": {
              "sha256": "4bd107465e68ec61a85e96b45fe1cdbdb914fa66674e2004c4a66335bb06a6e1",
              "bytes": 185688,
              "allBytesEqual": true
            },
            "albedo": {
              "sha256": "4bd107465e68ec61a85e96b45fe1cdbdb914fa66674e2004c4a66335bb06a6e1",
              "bytes": 185688,
              "allBytesEqual": true
            },
            "alpha": {
              "sha256": "97093a54b7457a011967609d8a227eb9e4b31bddc10a8f4b3b0901db8a3b4027",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "alphaPng": {
              "sha256": "9bea278d634d3ffe11997e7fc945daacd33c9c9a7a3d5d7a4344f1008243f1ad",
              "bytes": 13903,
              "allBytesEqual": true
            },
            "depth": {
              "sha256": "d01a20d6c70b74afa6b704e8cb88834fb4f836e1b93a10aef5238338d05289f8",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "depthPreview": {
              "sha256": "fea66789299727ad866556d5e42d96ea1d8e3211252c51f0cbfc09db9d1d9e0f",
              "bytes": 55657,
              "allBytesEqual": true
            },
            "depth512": {
              "sha256": "310afe59da126aade6fda51d2bf13d5063b4a0173b7528c2a98610519d648aa1",
              "bytes": 20654,
              "allBytesEqual": true
            }
          },
          "dependencies": [
            "colorMap RGB and alpha",
            "native depth",
            "fixed camera/affine",
            "sRGB conversion and fixed background"
          ],
          "method": "In-memory replay of unlit albedo/alpha/depth PNG and NPY encodings; no normal or lighting input"
        }
      },
      {
        "face": "+Y",
        "slug": "py",
        "sourceGuide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/color-prior/guides-unlit/py/guide.json",
          "sha256": "fad993fb181fa72220affa6df27d0024d3bbd79c64e0ef919747d4ffcae88be3",
          "bytes": 12989
        },
        "guide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/guides-unlit/py/guide.json",
          "sha256": "d4734297a09f883e62c0700658bb135d686b131fe0bf35eaa988a70f1cb7dc07",
          "bytes": 16323
        },
        "allImageRecordsUnchanged": true,
        "cameraMappingProxyUnchanged": true,
        "unlitReplay": {
          "normalMapReadByReplay": false,
          "allSevenFilesByteIdentical": true,
          "files": {
            "rgb": {
              "sha256": "b4b1c28a952cc1dcbab8b9116e641afb5a9298848ec877f7bc5b6a9fc2865a78",
              "bytes": 360688,
              "allBytesEqual": true
            },
            "albedo": {
              "sha256": "b4b1c28a952cc1dcbab8b9116e641afb5a9298848ec877f7bc5b6a9fc2865a78",
              "bytes": 360688,
              "allBytesEqual": true
            },
            "alpha": {
              "sha256": "0f9a8235f80a008a2bcc977bfebddc617c077553d8f255e2ce6104bb04ece058",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "alphaPng": {
              "sha256": "54ff5761b80170aed25db699e14ee14c37d0ae71287ea1283da148e6d786f330",
              "bytes": 16377,
              "allBytesEqual": true
            },
            "depth": {
              "sha256": "1ecfee654208fea91dceea1244d4b24f6f10387f4337737d67306d3604da1f54",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "depthPreview": {
              "sha256": "284087d83a8f4a2bac32066cca3d4cfafa4142a4559247584a11a878d477361c",
              "bytes": 81660,
              "allBytesEqual": true
            },
            "depth512": {
              "sha256": "7d50184d46827611826855cee805c331f72a63c095e9bca8e8de92866b078c70",
              "bytes": 30254,
              "allBytesEqual": true
            }
          },
          "dependencies": [
            "colorMap RGB and alpha",
            "native depth",
            "fixed camera/affine",
            "sRGB conversion and fixed background"
          ],
          "method": "In-memory replay of unlit albedo/alpha/depth PNG and NPY encodings; no normal or lighting input"
        }
      },
      {
        "face": "-Y",
        "slug": "ny",
        "sourceGuide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/color-prior/guides-unlit/ny/guide.json",
          "sha256": "040998e820c23290b1846a4d36f43b55960b8919dc227c5c2027ad05f64316d9",
          "bytes": 13028
        },
        "guide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/guides-unlit/ny/guide.json",
          "sha256": "7376289c93f432a78744f08dd6cc1994e0753a653e250507af2fc38301554831",
          "bytes": 16362
        },
        "allImageRecordsUnchanged": true,
        "cameraMappingProxyUnchanged": true,
        "unlitReplay": {
          "normalMapReadByReplay": false,
          "allSevenFilesByteIdentical": true,
          "files": {
            "rgb": {
              "sha256": "6b1736a740e7f35caa21018aaed3ee189728df6aedd75e1df0ac07a42758ab1a",
              "bytes": 342483,
              "allBytesEqual": true
            },
            "albedo": {
              "sha256": "6b1736a740e7f35caa21018aaed3ee189728df6aedd75e1df0ac07a42758ab1a",
              "bytes": 342483,
              "allBytesEqual": true
            },
            "alpha": {
              "sha256": "2ef87411a86a6cf1c53ebc0ca7e3b60274925c950848bba4e364703d0258d14e",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "alphaPng": {
              "sha256": "0a3e9296107137e8d25d7ab97fff319d8b929e12378ac8a5ddf6ad9edc250cad",
              "bytes": 15473,
              "allBytesEqual": true
            },
            "depth": {
              "sha256": "0f74304c62a4ebed442aa3ca8a71a4b7f0474aeffca9c77927e0ff37a5072eac",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "depthPreview": {
              "sha256": "59b1a4e1497b860d536e884e9aeabdf67f3552eb6021e4b1db10ae5fabfaf239",
              "bytes": 84854,
              "allBytesEqual": true
            },
            "depth512": {
              "sha256": "c07f869e055716a6787c9026e96c1af3b26be469359486605f83c7b2aeef1a9c",
              "bytes": 32925,
              "allBytesEqual": true
            }
          },
          "dependencies": [
            "colorMap RGB and alpha",
            "native depth",
            "fixed camera/affine",
            "sRGB conversion and fixed background"
          ],
          "method": "In-memory replay of unlit albedo/alpha/depth PNG and NPY encodings; no normal or lighting input"
        }
      },
      {
        "face": "+Z",
        "slug": "pz",
        "sourceGuide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/color-prior/guides-unlit/pz/guide.json",
          "sha256": "9d81e1abdfe48b41abe2b274de9cc65f2cacebd8157bbb767e16ed2bae6a9067",
          "bytes": 12944
        },
        "guide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/guides-unlit/pz/guide.json",
          "sha256": "42b41de819ea8828ce8e112d3d5dcdb89e1afe67e955b9b4bbc26aec9906d682",
          "bytes": 16278
        },
        "allImageRecordsUnchanged": true,
        "cameraMappingProxyUnchanged": true,
        "unlitReplay": {
          "normalMapReadByReplay": false,
          "allSevenFilesByteIdentical": true,
          "files": {
            "rgb": {
              "sha256": "e2462d3bf25d2f17ad533966360b08a7d6ddc298b0ef9b8a864e71c6dc384f65",
              "bytes": 184619,
              "allBytesEqual": true
            },
            "albedo": {
              "sha256": "e2462d3bf25d2f17ad533966360b08a7d6ddc298b0ef9b8a864e71c6dc384f65",
              "bytes": 184619,
              "allBytesEqual": true
            },
            "alpha": {
              "sha256": "ad3f795f520648374745b15a7435ecdcc67657a856feec26d137090d8a4568f1",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "alphaPng": {
              "sha256": "0b4215a71974bfa9696a26db1f9abcb27fb95c9ef039801782fa354ccc1075d7",
              "bytes": 14129,
              "allBytesEqual": true
            },
            "depth": {
              "sha256": "820e9475b2698821fe1f2faacd891cbdab54836bf240ad5f5d0f8737e6cba504",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "depthPreview": {
              "sha256": "764d1cae484e25a9c07c04c66d792747596828068ac83f397f8c5c63f1e195e0",
              "bytes": 58106,
              "allBytesEqual": true
            },
            "depth512": {
              "sha256": "01bcc73ed12b1a5501398bccbd4b6049f8b31ba0587ceb9093ae56615411ef08",
              "bytes": 21409,
              "allBytesEqual": true
            }
          },
          "dependencies": [
            "colorMap RGB and alpha",
            "native depth",
            "fixed camera/affine",
            "sRGB conversion and fixed background"
          ],
          "method": "In-memory replay of unlit albedo/alpha/depth PNG and NPY encodings; no normal or lighting input"
        }
      },
      {
        "face": "-Z",
        "slug": "nz",
        "sourceGuide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/color-prior/guides-unlit/nz/guide.json",
          "sha256": "b04e9bd204086b9ba418d7bb6af6bcfbe0c17514b42262e6051a669804b92140",
          "bytes": 12951
        },
        "guide": {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/guides-unlit/nz/guide.json",
          "sha256": "db812794eb36311557269f9623e5268e77e39e4ca513239260f864babe3049ac",
          "bytes": 16285
        },
        "allImageRecordsUnchanged": true,
        "cameraMappingProxyUnchanged": true,
        "unlitReplay": {
          "normalMapReadByReplay": false,
          "allSevenFilesByteIdentical": true,
          "files": {
            "rgb": {
              "sha256": "7bedc96579c907c168ce9e22a5dbac23e9dfe04d710fa96790b705132b327ec7",
              "bytes": 182629,
              "allBytesEqual": true
            },
            "albedo": {
              "sha256": "7bedc96579c907c168ce9e22a5dbac23e9dfe04d710fa96790b705132b327ec7",
              "bytes": 182629,
              "allBytesEqual": true
            },
            "alpha": {
              "sha256": "9ffa991d876b8234b830153f115790420915a0e41db4aa4c9e2212fd4ca7f90f",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "alphaPng": {
              "sha256": "ed36e23d44910a851c2f76af2207c4bd39c0b4c6c2e4174e2514b1b254c863b5",
              "bytes": 14183,
              "allBytesEqual": true
            },
            "depth": {
              "sha256": "f0d34d5f3157784f31d15a4199a41b34e65e9894d2e60657d88e06e421947e42",
              "bytes": 4194432,
              "allBytesEqual": true
            },
            "depthPreview": {
              "sha256": "d9beb9c72eba7d544ed797d9f3103299eebce62bd3b8c8acac0e31da05ca6eeb",
              "bytes": 55065,
              "allBytesEqual": true
            },
            "depth512": {
              "sha256": "08d4a37a4ec5acfea03cf11b4c64105a2bfeeb9c4f1b99bb319a5cc589383d76",
              "bytes": 20038,
              "allBytesEqual": true
            }
          },
          "dependencies": [
            "colorMap RGB and alpha",
            "native depth",
            "fixed camera/affine",
            "sRGB conversion and fixed background"
          ],
          "method": "In-memory replay of unlit albedo/alpha/depth PNG and NPY encodings; no normal or lighting input"
        }
      }
    ]
  },
  "allOriginalSourceFilesUnchanged": true,
  "qwenRunManifestsUnmodified": true,
  "seconds": 12.57732958800625
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/summary.json>>>
{
  "completed": true,
  "comparisonValid": true,
  "method": "official-inspector-rgb-only-box-240-v1",
  "selectedFaces": [
    "+Z",
    "+X",
    "-X",
    "-Z",
    "+Y",
    "-Y"
  ],
  "assetChecks": {
    "selectedFaces": [
      "+Z",
      "+X",
      "-X",
      "-Z",
      "+Y",
      "-Y"
    ],
    "unchangedFaces": [],
    "changedRgbPixels": 553065,
    "changedRgbPixelsByFace": {
      "+X": 97152,
      "-X": 97339,
      "+Y": 91973,
      "-Y": 91968,
      "+Z": 87323,
      "-Z": 87310
    },
    "colorAlphaBitDifferences": 0,
    "rgbaAlphaChecks": {
      "maps/color.rgba16f.bin": {
        "source": "4c7313a92d6a7a08db34e1bd427299af4637005a291698d8ae7447f9555abd03",
        "output": "4c7313a92d6a7a08db34e1bd427299af4637005a291698d8ae7447f9555abd03"
      },
      "maps/interval.rgba16f.bin": {
        "source": "f79263f5c2ec73570adf07119363a977941f23e05778a54348afaeeb8a059ed1",
        "output": "f79263f5c2ec73570adf07119363a977941f23e05778a54348afaeeb8a059ed1"
      },
      "maps/normal.rgba16f.bin": {
        "source": "4c7313a92d6a7a08db34e1bd427299af4637005a291698d8ae7447f9555abd03",
        "output": "4c7313a92d6a7a08db34e1bd427299af4637005a291698d8ae7447f9555abd03"
      },
      "maps/range.rgba16f.bin": {
        "source": "f77ca159ec58e2042669e3e88f50f17b4f63b40ed21098ffe1e41e6af5eb163a",
        "output": "f77ca159ec58e2042669e3e88f50f17b4f63b40ed21098ffe1e41e6af5eb163a"
      }
    },
    "unselectedFacesBitIdentical": true,
    "geometryMetadataSourceSha256": "5b40acc0cf5d082b87cb5b4f55ba255b1bcb99cd3411feb913d1c7e2fb4d0185",
    "geometryMetadataOutputSha256": "5b40acc0cf5d082b87cb5b4f55ba255b1bcb99cd3411feb913d1c7e2fb4d0185",
    "geometryDepthNormalIntervalRangeProxyCameraPreserved": true,
    "preservedSourceFiles": {
      "geometry/proxy.glb": "d7f5e2eb289a4565815782165486485cb9a161094695c1786141c81aea6064ee",
      "guides/nx.depth16f.bin": "59a8c6d4bcfc25ea1d4ded109b7fb8cc9634618b7a77a0903a0d46809607b8d2",
      "guides/nx.png": "5235cf0f4066375f46460e1273cb01bb18d669585d5296c8b7767696f5b9608d",
      "guides/ny.depth16f.bin": "c6c32bc26f4f0a119ad7c20507725e26fc80590c73b7d1721235f5ebc1592f95",
      "guides/ny.png": "253cbe1cf67f52c7a6a446c2058a76d84b5f0edb566bf3e7b74d20585acd537c",
      "guides/nz.depth16f.bin": "0d824bdc68140297532d555c152a4a132b4e925c7470fd849cca667fb84cf9d8",
      "guides/nz.png": "22c1ab7bb88008fbb1f7d48192bbc8b415d67eaca9defd1eb2136a911cc53fe4",
      "guides/px.depth16f.bin": "fd7d681090c1adb203f9eae512b7bf4b1ec8f7271edbaa91e35943937877a013",
      "guides/px.png": "0083969434908f9afd2d9f88558c22f40fa8fc27a195965449cc45e4048947b9",
      "guides/py.depth16f.bin": "bf96bf91e88e5eb9eca4c426eec95d2bf311a616742b367f4abfc16e80ca149f",
      "guides/py.png": "e58a0e96cbc0429e1b88a6adb38d95d4a987f1ca0d14089ca3a105abad3c2ee0",
      "guides/pz.depth16f.bin": "e92e21c5745e87b319478b79cee3ddc3a0970840300f12d659740a94643ce87d",
      "guides/pz.png": "f59c42ca6bd2029205fad2803edf016eeee1be2abfdfe3c10f43c5833263b8ba",
      "maps/interval.rgba16f.bin": "9458e5cda155b2d4e5b5482b229484abf0194f3d7e7fed8e40329f47f4dc2fd0",
      "maps/normal.rgba16f.bin": "bfe952bb631c95666f8054a2940cf28a0f3c721680549ee4f0d53987e0ed7741",
      "maps/range.rgba16f.bin": "f638fb30287fdffe24c4802d6fd2dd432b615918b00235d81fa577bfb5eeabb7",
      "qa/color-prior-13dcf86d0f78aab1.json": "109e3e9d5cdefaa5c196561451cd32c01eae3f7cd130de0e7e1d561462d35d94",
      "qa/normal-only-badb51598fbefcdc.json": "8d7af72eb0532221739c4e319c54647d3e39f0d0118f1162179bec972305d62a"
    },
    "sourceManifestSha256": "556b555829562c9cec35f97aa59e438eefdbd36f6619c4f1043b985c86568c87",
    "outputManifestSha256": "1ae379eb8b72a15c30d716c269c02b78352f0a1858b025c247efc5381c53eb13"
  },
  "receipt": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/variants/all6-smooth-prior-unlit-090/095fb922d6ace93bd4ae2b8679d2be1d91c5ee79dd1a2ef70d5b1580272272ee.json",
    "sha256": "650a43a5657cd717c3af2ab194acd9472b565925c7b737eddc6f2df2cd6bade1"
  },
  "inputHashes": {
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/smooth-normals/baselines/prior/manifest.json": "556b555829562c9cec35f97aa59e438eefdbd36f6619c4f1043b985c86568c87",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/variants/all6-smooth-prior-unlit-090/095fb922d6ace93bd4ae2b8679d2be1d91c5ee79dd1a2ef70d5b1580272272ee/manifest.json": "1ae379eb8b72a15c30d716c269c02b78352f0a1858b025c247efc5381c53eb13",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/variants/all6-smooth-prior-unlit-090/095fb922d6ace93bd4ae2b8679d2be1d91c5ee79dd1a2ef70d5b1580272272ee.json": "650a43a5657cd717c3af2ab194acd9472b565925c7b737eddc6f2df2cd6bade1",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/prepared/baseline512/manifest.json": "0fcd3e3d57f75aa35f8ec07af8541dca0c0dfe56167b49fef65f373533fd88ba",
    "/mnt/f/_dev/box-impostor-lab/experiments/zipsplat-qwen-20261120/capture_inspector.mjs": "a34b872abbad6b197b475692a06d3ade15681e52c55fd0b0c901d9202d7a1ab1",
    "/mnt/f/_dev/box-impostor-lab/experiments/zipsplat-qwen-20261120/compare_capture_alpha.py": "f85db40664b8e6259c52e9589cc2d01c662c76f3e25f2cd2969811d2ddd1347c",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/compare_boxes.py": "333099f104030663ad65bdb7a14b4d28cd8ca481661adc7a94f9627ce04b5b5e",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/compose_faces.py": "ca10a7373535664c3dd8b3a9289351a34e15865e2db0200f66e89a460d4faa1a",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/viewers/smooth-prior/smooth-prior512.inspector.html": "ba720adc3899f54f0851edbcd953b68db43ac7ed7ffcd7bfef50c01553d0c09b",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/viewers/all6-smooth-prior-unlit-090/all6-smooth-prior-unlit-090.inspector.html": "277075c56f8a2ec433b03607c140f2bdfeca19c2c63a2f38e5cdfd96e840a6e2"
  },
  "captures": {
    "before": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/before/capture.json",
      "sha256": "097772989ec079743c5c374a1c5eeeefa947790790a6fef925601318ffebec75"
    },
    "after": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/after/capture.json",
      "sha256": "08eb0877c877bf603ed580fdc596f2e7bcf1ae60f5dca27cce9f12f1b776a527"
    }
  },
  "captureEnvironmentEqual": true,
  "recordedCamerasExactlyEqual": true,
  "camera": {
    "maxAbsDifference": 0.0,
    "maxByField": {
      "position": 0.0,
      "quaternion": 0.0,
      "up": 0.0,
      "matrixWorld": 0.0,
      "matrixWorldInverse": 0.0,
      "projectionMatrix": 0.0,
      "projectionMatrixInverse": 0.0,
      "near": 0.0,
      "far": 0.0,
      "top": 0.0,
      "bottom": 0.0,
      "left": 0.0,
      "right": 0.0,
      "zoom": 0.0,
      "canvasPixels": 0.0
    },
    "differentFrameIndices": [],
    "verifiedFrames": 240,
    "differentFrames": 0
  },
  "alpha": {
    "framesCompared": 240,
    "exactFrames": 240,
    "changedFrames": 0,
    "allFramesZero": true,
    "all240FramesZero": true,
    "maeUint8": 0.0,
    "maxUint8": 0,
    "nonzeroPixels": 0,
    "nonzeroValues": 0,
    "sumAbsoluteDifference": 0,
    "values": 62914560,
    "pixels": 62914560
  },
  "rgb": {
    "framesCompared": 240,
    "exactFrames": 0,
    "changedFrames": 240,
    "allFramesZero": false,
    "all240FramesZero": false,
    "maeUint8": 0.9042599148220486,
    "maxUint8": 117,
    "nonzeroPixels": 5280731,
    "nonzeroValues": 15414756,
    "sumAbsoluteDifference": 170673344,
    "values": 188743680,
    "pixels": 62914560
  },
  "fullBody240Frames": {
    "before": [
      {
        "index": 0,
        "foregroundPixels": 20787,
        "boundsXYXY": [
          186,
          62,
          325,
          449
        ],
        "edgePixels": 0
      },
      {
        "index": 1,
        "foregroundPixels": 20693,
        "boundsXYXY": [
          187,
          62,
          323,
          449
        ],
        "edgePixels": 0
      },
      {
        "index": 2,
        "foregroundPixels": 20640,
        "boundsXYXY": [
          188,
          62,
          322,
          449
        ],
        "edgePixels": 0
      },
      {
        "index": 3,
        "foregroundPixels": 20625,
        "boundsXYXY": [
          188,
          62,
          321,
          449
        ],
        "edgePixels": 0
      },
      {
        "index": 4,
        "foregroundPixels": 20611,

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/encoding-summary.json>>>
{
  "completed": true,
  "specification": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/three-columns.json",
  "specification_sha256": "d6fb144b867e6da32fea1cddf4b1d9efbf2f43fff9538ee4f2bd3ce14f47af34",
  "script_sha256": "95a5242e6d01285f545f489b2284f67ff7e919d06a706ae4bf8dde912625ed64",
  "ffmpeg": "/mnt/f/_dev/box-impostor-lab/experiments/sdf-morph-20261120/tools/ffmpeg-linux-x86_64-v7.0.2",
  "encode_command": [
    "/mnt/f/_dev/box-impostor-lab/experiments/sdf-morph-20261120/tools/ffmpeg-linux-x86_64-v7.0.2",
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-f",
    "rawvideo",
    "-pixel_format",
    "rgb24",
    "-video_size",
    "1536x584",
    "-framerate",
    "24",
    "-i",
    "pipe:0",
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "19",
    "-pix_fmt",
    "yuv420p",
    "-frames:v",
    "240",
    "-movflags",
    "+faststart",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/comparison.partial.mp4"
  ],
  "layout": {
    "columns": 3,
    "rows": 1,
    "panel_pixels": [
      512,
      512
    ],
    "label_band_pixels": 36,
    "footer_pixels": 36,
    "output_pixels": [
      1536,
      584
    ],
    "background": "#808080",
    "compositing": "Pillow straight-sRGB alpha-over; identical for every panel",
    "resize": false,
    "crop": false
  },
  "frames": 240,
  "fps": 24,
  "duration_seconds": 10.0,
  "codec": "h264",
  "pixel_format": "yuv420p",
  "crf": 19,
  "faststart": true,
  "common_reference_sha256": "0fcd3e3d57f75aa35f8ec07af8541dca0c0dfe56167b49fef65f373533fd88ba",
  "validation": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/capture-validation.json",
    "sha256": "d20afb895844d8ab72f796fcfadf804a1469a4ad2c2f4730fdc850420f6d5807",
    "all240AlphaFramesEqualAcrossThreeColumns": true,
    "recordedCamerasExactlyEqual": true,
    "cameraTolerance": 0.0,
    "pairs": [
      {
        "columns": [
          0,
          1
        ],
        "labels": [
          "Gray Box",
          "Transferred color"
        ],
        "framesCompared": 240,
        "cameraMaxAbsDifference": 0.0,
        "cameraTolerance": 0.0,
        "alpha": {
          "all240FramesZero": true,
          "maxUint8": 0,
          "nonzeroPixels": 0
        },
        "rgb": {
          "changedFrames": 240,
          "maxUint8": 125,
          "nonzeroPixels": 5337104,
          "maeUint8": 2.294899559020996
        }
      },
      {
        "columns": [
          1,
          2
        ],
        "labels": [
          "Transferred color",
          "Transferred + Qwen"
        ],
        "framesCompared": 240,
        "cameraMaxAbsDifference": 0.0,
        "cameraTolerance": 0.0,
        "alpha": {
          "all240FramesZero": true,
          "maxUint8": 0,
          "nonzeroPixels": 0
        },
        "rgb": {
          "changedFrames": 240,
          "maxUint8": 117,
          "nonzeroPixels": 5280731,
          "maeUint8": 0.9042599148220486
        }
      },
      {
        "columns": [
          0,
          2
        ],
        "labels": [
          "Gray Box",
          "Transferred + Qwen"
        ],
        "framesCompared": 240,
        "cameraMaxAbsDifference": 0.0,
        "cameraTolerance": 0.0,
        "alpha": {
          "all240FramesZero": true,
          "maxUint8": 0,
          "nonzeroPixels": 0
        },
        "rgb": {
          "changedFrames": 240,
          "maxUint8": 122,
          "nonzeroPixels": 5334966,
          "maeUint8": 1.8530287636650933
        }
      }
    ]
  },
  "inputs": [
    {
      "label": "Gray Box",
      "capture": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/capture/smooth-gray/capture.json",
      "capture_sha256": "1f96cab636b04e89310d0b4684827a339c2507d3612b3e1a5b2d04874c0c336f",
      "html_sha256": "f90a3fce329b4ca9494b1b67be365008208c37c086ca3ceddf317e0623125b50",
      "verified_frame_hashes": 240,
      "verified_frame_order": true
    },
    {
      "label": "Transferred color",
      "capture": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/before/capture.json",
      "capture_sha256": "097772989ec079743c5c374a1c5eeeefa947790790a6fef925601318ffebec75",
      "html_sha256": "ba720adc3899f54f0851edbcd953b68db43ac7ed7ffcd7bfef50c01553d0c09b",
      "verified_frame_hashes": 240,
      "verified_frame_order": true
    },
    {
      "label": "Transferred + Qwen",
      "capture": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/after/capture.json",
      "capture_sha256": "08eb0877c877bf603ed580fdc596f2e7bcf1ae60f5dca27cce9f12f1b776a527",
      "html_sha256": "277075c56f8a2ec433b03607c140f2bdfeca19c2c63a2f38e5cdfd96e840a6e2",
      "verified_frame_hashes": 240,
      "verified_frame_order": true
    }
  ],
  "max_cross_capture_actual_yaw_difference_degrees": 0,
  "composite_rgb24_stream_sha256": "aa38b2245b61e166711b6f10a172e7af931eafacb6e429880e4cb5af0e236354",
  "decode": {
    "completed": true,
    "frames": 240,
    "rgb24_bytes": 645857280,
    "rgb24_stream_sha256": "2d12ef60f72ce8238aabbd441aff3a60a78839ba2aa58daac56f271cc2bfa3b7",
    "video_stream": "Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1536x584, 663 kb/s, 24 fps, 24 tbr, 12288 tbn (default)"
  },
  "artifacts": {
    "comparison.mp4": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/comparison.mp4",
      "sha256": "f1f87866d178de6238ccbfbf0c5a5542e603d38e7c63c01e0852ed61b657175a"
    },
    "poster-yaw000.png": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/poster-yaw000.png",
      "sha256": "6e28ae1395cdac1fa85927cdabe5e880e706e3c19ae836719afd50e794bbae54"
    },
    "poster-yaw090.png": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/poster-yaw090.png",
      "sha256": "f14311dbc126f7fcfc6016ae046a12908d52bb34f12421fe29482639b89c3eb6"
    },
    "poster-yaw180.png": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/poster-yaw180.png",
      "sha256": "28846160d77bb98e25494a962122f9fc9a5704b1f02df7be082f39700f58bb8c"
    },
    "poster-yaw270.png": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/poster-yaw270.png",
      "sha256": "c7e34189be345a43917ef08184dd4d7bd967807ec6907410a124a312386a5824"
    },
    "ffmpeg-encode.log": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/ffmpeg-encode.log",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    "ffmpeg-decode.log": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/ffmpeg-decode.log",
      "sha256": "afbfccd04474f51bdd6eb6da12f839f7e1154cad4adb40739ae3a3edbc6cf7d4"
    },
    "capture-validation.json": {
      "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/media/final-smooth-all6-090/capture-validation.json",
      "sha256": "d20afb895844d8ab72f796fcfadf804a1469a4ad2c2f4730fdc850420f6d5807"
    }
  }
}

<<<yomiyasu:material ファイル /mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/top-bottom/comparison.json>>>
{
  "completed": true,
  "method": "official-inspector-top-bottom-rgb-comparison-v1",
  "captures": {
    "before": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-before/capture.json",
    "after": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-after/capture.json"
  },
  "inputHashes": {
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-before/capture.json": "49d89d5b05aee7d3848157e543e0d8a4a2a223df34e0ea2efa60847661ba2a93",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/viewers/smooth-prior/smooth-prior512.inspector.html": "ba720adc3899f54f0851edbcd953b68db43ac7ed7ffcd7bfef50c01553d0c09b",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/prepared/baseline512/manifest.json": "0fcd3e3d57f75aa35f8ec07af8541dca0c0dfe56167b49fef65f373533fd88ba",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-before/top.png": "1cab6cbd973bcb081d291dd5c7f25763122599ecf6c2ce49b8e2a5aede5296f0",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-before/bottom.png": "afb8d379d301e2e0c8276c7e20d059fffd18fa8288303b2930bd122c7b4284cc",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-after/capture.json": "cfe2fac6e567d783960a11e797be6f2acf284746df6b352b168336e8547ed46e",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/viewers/all6-smooth-prior-unlit-090/all6-smooth-prior-unlit-090.inspector.html": "277075c56f8a2ec433b03607c140f2bdfeca19c2c63a2f38e5cdfd96e840a6e2",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-after/top.png": "47542f48e5c4525ca3c33df40728ae29a1d23c37b3646e369f39a7c4b0295de6",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-after/bottom.png": "8fbd19c05a54807586953a4174c4569085645d51577ca738ab7ccb7112460899",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/compare_top_bottom.py": "7ed122c179352c0779bf33158bf3479767d808fa965bc3e05bc7aa54ae658c91",
    "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/scripts/encode_three.py": "95a5242e6d01285f545f489b2284f67ff7e919d06a706ae4bf8dde912625ed64"
  },
  "cameraMaxAbsDifference": 0,
  "all2AlphaFramesZero": true,
  "alphaMaxUint8": 0,
  "alphaNonzeroPixels": 0,
  "allFramesNonemptyAndUnclipped": true,
  "htmlUnmodified": true,
  "sheet": {
    "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/top-bottom/top-bottom-before-after.png",
    "sha256": "c1496f451e9eca8cdad42572418ab3d2268360e797d69a243f683561b6a55324",
    "pixels": [
      1024,
      1096
    ],
    "sourceFramePixels": [
      512,
      512
    ],
    "crop": false,
    "resize": false,
    "backgroundRGB": [
      128,
      128,
      128
    ],
    "titleHeight": 36
  },
  "views": [
    {
      "view": "top",
      "sources": [
        {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-before/top.png",
          "sha256": "1cab6cbd973bcb081d291dd5c7f25763122599ecf6c2ce49b8e2a5aede5296f0",
          "camera": {
            "type": "OrthographicCamera",
            "position": [
              -0.00035370886325836176,
              1.8039003268619394,
              -0.0031031895679428426
            ],
            "quaternion": [
              -0.7071064276330686,
              0,
              0,
              0.7071071347398497
            ],
            "up": [
              0,
              1,
              0
            ],
            "matrixWorld": [
              1,
              0,
              0,
              0,
              0,
              9.999999996956888e-07,
              -0.9999999999995001,
              0,
              0,
              0.9999999999995001,
              9.999999996956888e-07,
              0,
              -0.00035370886325836176,
              1.8039003268619394,
              -0.0031031895679428426,
              1
            ],
            "matrixWorldInverse": [
              1,
              0,
              0,
              0,
              0,
              9.999999996956885e-07,
              0.9999999999994998,
              0,
              0,
              -0.9999999999994998,
              9.999999996956885e-07,
              0,
              0.00035370886325836176,
              -0.0031049934682676034,
              -1.8039003237578477,
              1
            ],
            "projectionMatrix": [
              1.524138466721586,
              0,
              0,
              0,
              0,
              1.524138466721586,
              0,
              0,
              0,
              0,
              -0.028384436410031102,
              0,
              0,
              0,
              -1.0000166668055566,
              1
            ],
            "projectionMatrixInverse": [
              0.6561083666833728,
              0,
              0,
              0,
              0,
              0.6561083666833728,
              0,
              0,
              0,
              0,
              -35.23057444419078,
              0,
              0,
              0,
              -35.23116162532469,
              1
            ],
            "near": 0.0005871811339126289,
            "far": 70.46173606951547,
            "top": 0.6561083666833728,
            "bottom": -0.6561083666833728,
            "left": -0.6561083666833728,
            "right": 0.6561083666833728,
            "zoom": 1,
            "canvasPixels": [
              512,
              512
            ]
          },
          "actualOutward": [
            0,
            0.9999999999995001,
            9.999999996956888e-07
          ],
          "axisErrorDegrees": 5.729196509083288e-05
        },
        {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-after/top.png",
          "sha256": "47542f48e5c4525ca3c33df40728ae29a1d23c37b3646e369f39a7c4b0295de6",
          "camera": {
            "type": "OrthographicCamera",
            "position": [
              -0.00035370886325836176,
              1.8039003268619394,
              -0.0031031895679428426
            ],
            "quaternion": [
              -0.7071064276330686,
              0,
              0,
              0.7071071347398497
            ],
            "up": [
              0,
              1,
              0
            ],
            "matrixWorld": [
              1,
              0,
              0,
              0,
              0,
              9.999999996956888e-07,
              -0.9999999999995001,
              0,
              0,
              0.9999999999995001,
              9.999999996956888e-07,
              0,
              -0.00035370886325836176,
              1.8039003268619394,
              -0.0031031895679428426,
              1
            ],
            "matrixWorldInverse": [
              1,
              0,
              0,
              0,
              0,
              9.999999996956885e-07,
              0.9999999999994998,
              0,
              0,
              -0.9999999999994998,
              9.999999996956885e-07,
              0,
              0.00035370886325836176,
              -0.0031049934682676034,
              -1.8039003237578477,
              1
            ],
            "projectionMatrix": [
              1.524138466721586,
              0,
              0,
              0,
              0,
              1.524138466721586,
              0,
              0,
              0,
              0,
              -0.028384436410031102,
              0,
              0,
              0,
              -1.0000166668055566,
              1
            ],
            "projectionMatrixInverse": [
              0.6561083666833728,
              0,
              0,
              0,
              0,
              0.6561083666833728,
              0,
              0,
              0,
              0,
              -35.23057444419078,
              0,
              0,
              0,
              -35.23116162532469,
              1
            ],
            "near": 0.0005871811339126289,
            "far": 70.46173606951547,
            "top": 0.6561083666833728,
            "bottom": -0.6561083666833728,
            "left": -0.6561083666833728,
            "right": 0.6561083666833728,
            "zoom": 1,
            "canvasPixels": [
              512,
              512
            ]
          },
          "actualOutward": [
            0,
            0.9999999999995001,
            9.999999996956888e-07
          ],
          "axisErrorDegrees": 5.729196509083288e-05
        }
      ],
      "cameraMaxAbsDifference": 0.0,
      "cameraFieldDifferences": {
        "position": 0.0,
        "quaternion": 0.0,
        "up": 0.0,
        "matrixWorld": 0.0,
        "matrixWorldInverse": 0.0,
        "projectionMatrix": 0.0,
        "projectionMatrixInverse": 0.0,
        "near": 0.0,
        "far": 0.0,
        "top": 0.0,
        "bottom": 0.0,
        "left": 0.0,
        "right": 0.0,
        "zoom": 0.0,
        "canvasPixels": 0.0
      },
      "alpha": {
        "maxUint8": 0,
        "nonzeroPixels": 0
      },
      "rgb": {
        "maxUint8": 98,
        "nonzeroPixels": 8090,
        "sumAbsoluteDifference": 277301,
        "values": 786432
      },
      "coverage": [
        {
          "foregroundPixels": 8142,
          "edgePixels": 0,
          "boundsXYXY": [
            187,
            189,
            324,
            322
          ]
        },
        {
          "foregroundPixels": 8142,
          "edgePixels": 0,
          "boundsXYXY": [
            187,
            189,
            324,
            322
          ]
        }
      ]
    },
    {
      "view": "bottom",
      "sources": [
        {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-before/bottom.png",
          "sha256": "afb8d379d301e2e0c8276c7e20d059fffd18fa8288303b2930bd122c7b4284cc",
          "camera": {
            "type": "OrthographicCamera",
            "position": [
              -0.00035370886325836176,
              -1.8013855174918982,
              -0.00310318956794237
            ],
            "quaternion": [
              0.7071064276330684,
              0,
              0,
              0.7071071347398499
            ],
            "up": [
              0,
              1,
              0
            ],
            "matrixWorld": [
              1,
              0,
              0,
              0,
              0,
              1.0000000003618226e-06,
              0.9999999999995001,
              0,
              0,
              -0.9999999999995001,
              1.0000000003618226e-06,
              0,
              -0.00035370886325836176,
              -1.8013855174918982,
              -0.00310318956794237,
              1
            ],
            "matrixWorldInverse": [
              1,
              0,
              0,
              0,
              0,
              1.0000000003618224e-06,
              -0.9999999999994998,
              0,
              0,
              0.9999999999994998,
              1.0000000003618224e-06,
              0,
              0.00035370886325836176,
              0.0031049909534589613,
              -1.8013855143878077,
              1
            ],
            "projectionMatrix": [
              1.524138466721586,
              0,
              0,
              0,
              0,
              1.524138466721586,
              0,
              0,
              0,
              0,
              -0.028384436410031102,
              0,
              0,
              0,
              -1.0000166668055566,
              1
            ],
            "projectionMatrixInverse": [
              0.6561083666833728,
              0,
              0,
              0,
              0,
              0.6561083666833728,
              0,
              0,
              0,
              0,
              -35.23057444419078,
              0,
              0,
              0,
              -35.23116162532469,
              1
            ],
            "near": 0.0005871811339126289,
            "far": 70.46173606951547,
            "top": 0.6561083666833728,
            "bottom": -0.6561083666833728,
            "left": -0.6561083666833728,
            "right": 0.6561083666833728,
            "zoom": 1,
            "canvasPixels": [
              512,
              512
            ]
          },
          "actualOutward": [
            0,
            -0.9999999999995001,
            1.0000000003618226e-06
          ],
          "axisErrorDegrees": 5.729196509083288e-05
        },
        {
          "path": "/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121/comparison/final-smooth-all6-090/capture/top-bottom-after/bottom.png",
          "sha256": "8fbd19c05a54807586953a4174c4569085645d51577ca738ab7ccb7112460899",
          "camera": {
            "type": "OrthographicCamera",
            "position": [
              -0.00035370886325836176,
              -1.8013855174918982,
              -0.00310318956794237
            ],
            "quaternion": [
              0.7071064276330684,
              0,
              0,
              0.7071071347398499
            ],
            "up": [
              0,
              1,
              0
            ],
            "matrixWorld": [
              1,
              0,
              0,
              0,
              0,
              1.0000000003618226e-06,
              0.9999999999995001,
              0,
              0,
              -0.9999999999995001,
              1.0000000003618226e-06,
              0,
              -0.00035370886325836176,
              -1.8013855174918982,
              -0.00310318956794237,
              1
            ],
            "matrixWorldInverse": [
              1,
              0,
              0,
              0,
              0,
              1.0000000003618224e-06,
              -0.9999999999994998,
              0,
              0,
              0.9999999999994998,
              1.0000000003618224e-06,
              0,
              0.00035370886325836176,
              0.0031049909534589613,
              -1.8013855143878077,
              1
            ],
            "projectionMatrix": [
              1.524138466721586,
              0,
              0,
              0,
              0,
              1.524138466721586,
              0,
              0,
              0,
              0,
              -0.028384436410031102,
              0,
              0,
              0,
              -1.0000166668055566,
              1
            ],
            "projectionMatrixInverse": [
              0.6561083666833728,
              0,
              0,
              0,
              0,
              0.6561083666833728,
              0,
              0,
              0,
              0,
              -35.23057444419078,
              0,
              0,
              0,
              -35.23116162532469,
              1
            ],
            "near": 0.0005871811339126289,
            "far": 70.46173606951547,
            "top": 0.6561083666833728,
            "bottom": -0.6561083666833728,
            "left": -0.6561083666833728,
            "right": 0.6561083666833728,
            "zoom": 1,
            "canvasPixels": [
              512,
              512
            ]
          },
          "actualOutward": [
            0,
            -0.9999999999995001,
            1.0000000003618226e-06
          ],
          "axisErrorDegrees": 5.729196509083288e-05
        }
      ],
      "cameraMaxAbsDifference": 0.0,
      "cameraFieldDifferences": {
        "position": 0.0,
        "quaternion": 0.0,
        "up": 0.0,
        "matrixWorld": 0.0,
        "matrixWorldInverse": 0.0,
        "projectionMatrix": 0.0,
        "projectionMatrixInverse": 0.0,
        "near": 0.0,
        "far": 0.0,
        "top": 0.0,
        "bottom": 0.0,
        "left": 0.0,
        "right": 0.0,
        "zoom": 0.0,
        "canvasPixels": 0.0
      },
      "alpha": {
        "maxUint8": 0,
        "nonzeroPixels": 0
      },
      "rgb": {
        "maxUint8": 71,
        "nonzeroPixels": 7610,
        "sumAbsoluteDifference": 132936,
        "values": 786432
      },
      "coverage": [
        {
          "foregroundPixels": 8143,
          "edgePixels": 0,
          "boundsXYXY": [
            187,
            189,
            324,
            322
          ]
        },
        {
          "foregroundPixels": 8143,
          "edgePixels": 0,
          "boundsXYXY": [
            187,
            189,
            324,
            322
          ]
        }
      ]
    }
  ],
  "limitations": [
    "Alpha equality is limited to these two cameras and 8-bit PNG precision.",
    "RGB difference proves a change, not internal texture correspondence or quality."
  ]
}


