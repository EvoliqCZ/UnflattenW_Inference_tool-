# Unflatten W — current product facts

The current release is **Unflatten W 1.0**, using **Unflatten W Dense 1.0**,
by **EVOLIQ s.r.o.** The authoritative release facts, evaluation methodology,
limitations, licensing, release notes and ready-to-use Slovak copy are in
[RELEASE.md](../RELEASE.md). Its publication-status note applies to these links.

Unflatten W reconstructs editable Word documents from PDF, PNG and JPEG inputs.
The Linux/NVIDIA application offers an English/Slovak interface, conversion jobs,
downloads, available previews and model management. Integrations can use the
[authenticated API and local Swagger](../API.md) without a browser login.

The released model uses a frozen Qwen3-VL-4B-Instruct backbone with trained Dense
document modules and typed property heads. Standard deployment uses BF16 and
recommends at least 16 GB GPU memory, subject to available-memory checks. The
separate local FP8 test bundle is not distributed. MoE, MTP and DSpark are not
features of this release.

Model-quality evidence is a diagnostic comparison on **100 fixed validation
pages**, not an independent test set. Word-file creation and structural validity
do not establish text accuracy or visual fidelity. Generated documents require
review. UI language support does not imply a model language-quality guarantee.

EVOLIQ's original contributions are free for private, noncommercial personal use;
professional, organizational and commercial use requires a paid written agreement.
Third-party rights, including the Qwen backbone's Apache-2.0 licence, remain unchanged.
See the [licence](../../LICENSE) and [release facts](../RELEASE.md#licensing).

For visual materials, use the supplied product mark and the original company
artwork described in [EVOLIQ-ASSETS.md](EVOLIQ-ASSETS.md), with synthetic demo
content. Do not reuse historical model timings, parameter counts or architecture
roadmaps as claims about Dense 1.0.

[Model](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0) ·
[Application](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-) ·
[EVOLIQ s.r.o.](https://www.evoliq.cz/sk)
