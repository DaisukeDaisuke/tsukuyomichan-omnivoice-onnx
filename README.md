# tsukuyomichan-omnivoice-onnx

`kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune` を、ブラウザ/ONNX Runtime向けのsplit ONNXへ変換し、FP32品質baselineとMobile INT8 weight-only版をGitHub Release / Hugging Face mirrorへ公開するための変換専用リポジトリです。

| 配布profile | Hugging Face直リンク | GitHub Release | 量子化 |
| --- | --- | --- | --- |
| Mobile INT8 | [mobile-int8](https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/tree/mobile-int8) | `mobile-int8-latest` | LLMの定数MatMul weightのみ8-bit。activation / audio embeddings / audio heads / Higgs decoderはFP32 |
| FP32 baseline | [main](https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/tree/main) | `full-finetune-latest` | なし |

Mobile INT8は実試聴で実用上ほぼ劣化なしと確認済みです。わずかに音の豊かさが減った可能性はありますが、試聴上の差は小さく、ブラウザ向けの標準候補として扱います。

## 変換方針

GitHub Actionsの `Convert full-finetune ONNX profiles` は `main` へのpush時に、固定revisionのfull-finetuneをrunnerへ取得して **FP32とMobile INT8をどちらも無条件でbuild** します。手動の `workflow_dispatch` でも同じ2 profileをbuildします。

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

FP32 profileはLLM、audio embedding/head、Higgs decoderをすべてFP32で維持し、量子化operatorまたは低精度weight initializerが1個でも存在した場合はRelease gateで拒否します。Mobile INT8 profileは、この検証済みFP32 graphから `llm_decoder.onnx` の定数 `MatMul` weightだけを8-bit `com.microsoft::MatMulNBits`へ変換します。activation、audio embeddings、audio heads、Higgs decoderはFP32のままです。量子化前後のgolden比較とgraph contract検査を通過したものだけをReleaseします。

OmniVoiceのiterative unmaskingはautoregressive causal decodingではありません。LLM backboneは元実装と同じ`[batch, 1, sequence, sequence]`の4次元Boolean attention maskを受ける形で直接ONNX exportし、KV cacheは使用しません。conditional sequenceは全位置を相互参照し、unconditional sequenceも実target区間を全結合にします。2次元padding/causal maskへ変換されたLLMはRelease gateで拒否します。

つくよみちゃんfull-finetune向け推論設定は `num_step=16` を採用し、それ以外はOmniVoiceの標準生成値（`guidance_scale=2.0`、`t_shift=0.1`、`layer_penalty_factor=5.0`、`position_temperature=5.0`、`class_temperature=0.0`）を `runtime-manifest.json` に記録します。

## Source checkpointをArtifactへ保存しない

元の `model.safetensors`（2,450,344,144 bytes）は**runner-localの一時build inputだけ**です。`actions/upload-artifact`、Actions cache、GitHub Releaseのいずれにも保存しません。backbone exportに必要な情報を作成した時点で削除します。Higgs Audio 2の元safetensorsも同様にReleaseへ含めません。

Workflowには `actions/upload-artifact` と `actions/cache` を使用していません。Release直前にも `.safetensors` が存在しないことを再検査します。

## Release

`main` pushまたは `workflow_dispatch` 成功時に、FP32は `full-finetune-latest`、Mobile INT8は `mobile-int8-latest` Releaseをそれぞれ更新します。Release assetは1ファイル1.8 GB以下へ制限し、大きなONNX external dataはtensor境界を保ったまま複数ファイルへ分割します。

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

`runtime-manifest.json` は各runtime assetについて SHA-256 に加えて XXH3-128 も記録します。SHA-256はCI/Release監査用に維持しますが、iPadを含むブラウザruntimeはmulti-GB assetにSHA-256を計算せず、初回取得・再ロードともXXH3-128で検証します。これはブラウザ側の改ざん検出を非暗号学的ハッシュへ弱める意図的な性能上のトレードオフです。配布元はimmutable Hugging Face revisionへ固定します。

Hugging Face側のModel Card metadataは `base_model: kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune` とし、変換モデルの親をfull-finetune checkpointへ直接結びます。FP32はHugging Face `main`、Mobile INT8は `mobile-int8` branchへ公開し、各buildは `gh-<GitHub SHA>-<GitHub run ID>-<profile>` というbuild固有tagも作ります。`runtime-manifest.json`にもその固定revisionを記録するため、ブラウザruntimeはmutable branchではなくbuild固有revisionへ固定できます。

Workflowにはrepository secret **`HF_TOKEN`** が必須です。Hugging FaceのFine-grained Access Tokenの `CI/CD` presetを使用し、token値をリポジトリへcommitしないでください。元の2.45 GB voice checkpointやHiggs source checkpointはHugging Face mirrorにもアップロードしません。

## Audio Samples

以下のWAVは、各profileのRelease workflowで変換・検証が完了したruntimeを使い、GitHub ActionsのCPU runner上でnative Python + ONNX Runtimeにより**同一文章・同一seed**でオフライン生成します。下のplayerは生成済みWAVを再生するだけで、ブラウザ内でモデル推論は行いません。FP32とMobile INT8をそのまま聞き比べられます。

### Japanese: 税関関税許可局

Text: `税関関税許可局、関税許可を急遽却下`

FP32:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/main/samples/01_customs_tariff_rejection.wav"></audio>

Mobile INT8:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/mobile-int8/samples/01_customs_tariff_rejection.wav"></audio>

### Japanese: WebAssembly

Text: `WebAssemblyをLLMでVibe Coding中`

FP32:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/main/samples/02_webassembly_vibe_coding.wav"></audio>

Mobile INT8:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/mobile-int8/samples/02_webassembly_vibe_coding.wav"></audio>

### Japanese: えへへ、見つけてくれたんだ！

Text: `えへへ、見つけてくれたんだ！ずっとここで待ってたんだよ？`

FP32:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/main/samples/03_found_me_waiting.wav"></audio>

Mobile INT8:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/mobile-int8/samples/03_found_me_waiting.wav"></audio>

### English

Text: `Hey, you finally made it! How does it feel, looking back at everything we've been through?`

FP32:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/main/samples/04_found_me_waiting_English.wav"></audio>

Mobile INT8:
<audio controls src="https://huggingface.co/RabbitDaisuke/tsukuyomichan-omnivoice-full-finetune-onnx/resolve/mobile-int8/samples/04_found_me_waiting_English.wav"></audio>

# Download and try your computer!

ブラウザPoCは次から直接試せます。

https://daisukedaisuke.github.io/typed-voice/poc.html

PoCは変換済みFP32 runtimeをブラウザの永続Cacheへ保存して、reload後も再利用します。現在のruntimeは約**2.4 GiB**あるため、初回準備では2 GBを超える通信量とローカルストレージを使用します。回線容量と空きストレージを確認してから実行してください。容量を戻したい場合は、このサイトの保存済みデータを削除すれば永続モデルCacheも削除できます。

モデルassetのoffline path、永続Cache、cross-origin isolationには**Service Worker**を使用します。WebGPUを利用できるdesktop browserでは、音質検証済みの**WebGPU + WebAssembly hybrid inference**を使用します。audio embeddings / LLM / audio headsはWebGPU、Higgs waveform decoderはWebAssemblyで実行します。Higgs decoderをWebGPUで実行すると音質劣化が確認されたため、ここだけWebAssemblyへ固定しています。必要なWebGPU経路を利用できないbrowserでは、対応可能な場合はWebAssemblyへfallbackします。

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

このモデルは、**つくよみちゃんというキャラクターの声をもとにした音声合成モデルです。**
モデルの変換形式や配布形態が変わっても、誰の声をもとにしているのか分からなくならないよう、つくよみちゃん由来であることを明記しています。

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

通常は `main` へのpushでFP32 / Mobile INT8の両profileを自動buildします。手動実行する場合はGitHubの `Actions` → `Convert full-finetune ONNX profiles` → `Run workflow` を使用します。正常終了したprofileだけが対応する `full-finetune-latest` / `mobile-int8-latest` の既存assetとHugging Face branchを更新します。