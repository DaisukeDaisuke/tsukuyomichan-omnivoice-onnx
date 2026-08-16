# tsukuyomichan-omnivoice-onnx

`kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune` を、ブラウザ/ONNX Runtime向けのsplit ONNXへ**非量子化FP32のまま**変換し、GitHub ReleaseとHugging Face mirrorへ公開するための変換専用リポジトリです。

## 変換方針

GitHub Actionsの `Convert full-finetune to FP32 ONNX` を `workflow_dispatch` するたびに、固定revisionのfull-finetuneをrunnerへ取得し、次のruntimeへ変換します。

```text
audio_embeddings_encoder.onnx
        ↓
llm_decoder.onnx
        ↓
audio_heads_decoder.onnx
        ↓ iterative unmasking (runtime側)
higgs_decoder.onnx
        ↓
24 kHz waveform
```

変換ではINT4、INT8、GPTQ、FP16、BF16を使用しません。LLM、audio embedding/head、Higgs decoderはいずれもFP32です。Release前にONNX graphを検査し、量子化operatorまたは低精度weight initializerが1個でも存在した場合はWorkflowを失敗させます。また、exportしたcomponentはPyTorch出力とONNX Runtime出力を数値比較してからReleaseへ進みます。

OmniVoiceのiterative unmaskingはautoregressive causal decodingではありません。LLM backboneは元実装と同じ`[batch, 1, sequence, sequence]`の4次元Boolean attention maskを受ける形で直接ONNX exportし、KV cacheは使用しません。conditional sequenceは全位置を相互参照し、unconditional sequenceも実target区間を全結合にします。2次元padding/causal maskへ変換されたLLMはRelease gateで拒否します。

つくよみちゃんfull-finetune向け推論設定は `num_step=16` を採用し、それ以外はOmniVoiceの標準生成値（`guidance_scale=2.0`、`t_shift=0.1`、`layer_penalty_factor=5.0`、`position_temperature=5.0`、`class_temperature=0.0`）を `runtime-manifest.json` に記録します。

## Source checkpointをArtifactへ保存しない

元の `model.safetensors`（2,450,344,144 bytes）は**runner-localの一時build inputだけ**です。`actions/upload-artifact`、Actions cache、GitHub Releaseのいずれにも保存しません。backbone exportに必要な情報を作成した時点で削除します。Higgs Audio 2の元safetensorsも同様にReleaseへ含めません。

Workflowには `actions/upload-artifact` と `actions/cache` を使用していません。Release直前にも `.safetensors` が存在しないことを再検査します。

## Release

`workflow_dispatch` 成功時に `full-finetune-latest` Releaseを更新します。Release assetは1ファイル1.8 GB以下へ制限し、大きなONNX external dataはtensor境界を保ったまま複数ファイルへ分割します。

主な配布物:

- `audio_embeddings_encoder.onnx` + external data
- `llm_decoder.onnx` + external data
- `audio_heads_decoder.onnx` + external data
- `higgs_decoder.onnx` + external data
- full-finetune由来tokenizer/config
- `runtime-manifest.json`
- `SHA256SUMS`
- `NOTICE.txt`
- つくよみちゃんfull-finetuneのmodel card
- pinned `k2-fsa/OmniVoice` のmodel card / code LICENSE
- Boson Higgs Audio 2 / Meta Llama 3のライセンス原文

## Hugging Face browser mirror

GitHub Releaseは監査・Releaseアーカイブとして維持します。同じ検証済みruntime assetを、ブラウザからCORS/Range取得するためのmirrorとして `RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx` にもGitHub Actionsからアップロードします。

Hugging Face側のModel Card metadataは `base_model: kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune` とし、変換モデルの親をfull-finetune checkpointへ直接結びます。各Workflow runは `gh-<GitHub SHA>-<GitHub run ID>` というbuild固有tagを作り、`runtime-manifest.json`にもその固定revisionを記録します。ブラウザruntimeはmutableな `main` ではなく、このbuild固有revisionとasset SHA-256を使用できます。

Workflowにはrepository secret **`HF_TOKEN`** が必須です。Hugging FaceのFine-grained Access Tokenの `CI/CD` presetを使用し、token値をリポジトリへcommitしないでください。元の2.45 GB voice checkpointやHiggs source checkpointはHugging Face mirrorにもアップロードしません。

## Source pins

- Voice: `kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune` @ `c1d7ff9477d0b21f220c58070da63355f69607e9`
- Voice `model.safetensors` SHA-256: `9ebaa8dd3bf35ceb6217cd19142bdabe6d6c044cca40672d2ae163d1a90ab47e`
- Higgs codec source: `k2-fsa/OmniVoice` @ `5337ba6bfe0ab30725fed141678a054fbedbf7da`
- Higgs `audio_tokenizer/model.safetensors` SHA-256: `fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d`
- OmniVoice converter/runtime code license source: `k2-fsa/OmniVoice` release `0.2.1` @ `5ba967c4d5b0f08244ae856b033eea583d1e4517`

model checkpointは固定revisionに加えてbyte sizeとSHA-256を検証します。OmniVoice code LICENSEは`omnivoice==0.2.1`のimmutable Git commitから取得し、Meta Llama 3 LICENSEは取得内容のSHA-256を検証します。

## ライセンスと利用条件

このリポジトリにある**変換コード**はトップレベル `LICENSE` のMIT Licenseです。変換後モデル、元モデル、つくよみちゃんコーパス由来部分、Higgs Audio 2、Meta Llama 3がMITへ再ライセンスされるわけではありません。

本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

■つくよみちゃんコーパス（CV.夢前黎）
https://tyc.rei-yumesaki.net/material/corpus/

full-finetuneのmodel cardが定める生成音声の利用制限、改変・再配布条件、つくよみちゃんコーパス利用条件を確認してください。WorkflowはReleaseへ英語/日本語model cardを同梱します。

また、初期checkpointである `k2-fsa/OmniVoice` の公式model cardは、コードをApache-2.0、pre-trained modelを学習データ制約によりCC-BY-NCと記載しています。この変換リポジトリは、それらのupstream model条件がfull-finetuneやONNX化によって消滅・緩和されたとは扱いません。固定model revisionのOmniVoice model cardに加え、変換で使用する `omnivoice==0.2.1` に対応するGitHub release commitからcode LICENSEを別途取得してReleaseへ同梱します。Hugging Face model repoのrootにはcode LICENSEが存在しないため、`audio_tokenizer/LICENSE` と混同しません。

Higgs decoderはBoson Higgs Audio 2 materialsを利用します。ReleaseにはBoson Higgs Audio 2 Community License AgreementとMeta Llama 3 Community License Agreementのコピー、および必要なNOTICEを同梱します。

Boson Higgs Audio 2 is licensed under the Boson Community License, Copyright © Boson AI USA, Inc. All Rights Reserved.

Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

Built with Higgs Materials licensed from Boson AI USA, Inc.

Built with Meta Llama 3.

## 手動実行

GitHubの `Actions` → `Convert full-finetune to FP32 ONNX` → `Run workflow` から実行します。正常終了したrunだけが `full-finetune-latest` の既存assetを置き換えます。