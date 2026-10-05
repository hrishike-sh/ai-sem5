# ViolationLens demo

Start the app with `Start.cmd` and open `http://127.0.0.1:8521`. If setup is incomplete, follow the project README, resolve the items shown in the app, and choose **Recheck setup**. Opening a saved run does not require starting detection again.

Start with **Look at the image with Astra**, the main workflow. Choose a permitted image, acknowledge permission, and select **Upload and preview**. A video can also supply one frame at a selected timestamp. Inspect the exact image, acknowledge that it will be sent to OpenAI using your ChatGPT plan, and select **Send this image to Astra**. Read its observations, suspected concerns, evidence references and uncertainty. Save a reviewer disposition, export the JSON, then reload: the saved assessment should return without another request. The model is explicitly `gpt-6-astra`; unavailable access is reported rather than silently switching models. Detector setup is unnecessary for this path.

A live direct Astra request completed on a clearly synthetic rider illustration, returning a rider observation and possible missing-helmet concern. This verifies the image protocol and structured result, not real-world accuracy. Sending the previously prepared real-traffic images remains blocked pending specific sharing approval. Real clips can still be processed and inspected locally.

The following geometric and detector demonstrations are under **Optional local detection and rule review**:

For the shortest repeatable walkthrough, expand **Try a synthetic rule demo** in the source panel. Choose **Opposing direction - positive rule case** and open the demo. This generates a three-second labeled rectangle animation and passes its known track coordinates through the existing wrong-way rule. It produces one supported configured-rule candidate at 0.50 seconds, with evidence at 0.00, 0.50 and 1.00 seconds. Inspect those frames, add a note saying this is a synthetic configuration test, accept the candidate, export, and reload to confirm it stays saved. No detector, model weights or ChatGPT connection is needed for these steps.

Then try **Following direction - negative rule case**: the engine records `no_violation_observed`, with zero supported positive candidates. Finally try **Direction not configured - unknown case**: the engine records `cannot_assess` and the missing direction context remains visible. Each choice creates a separate saved run. All generated frames and run records identify synthetic evidence. These demonstrations test geometry and the review workflow, not learned-model accuracy or legal violations. Optional explanations keep that distinction and cannot change the deterministic result.

For a real clip, follow this flow:

1. Choose a short permitted MP4, MOV or AVI clip, acknowledge your permission to analyze it, and select **Upload clip**.
2. Select **Run detection**. Follow the progress message. Cancellation asks the worker to stop; wait for the saved status to change. Reloading the page restores the current run, and **Continue a review** can reopen earlier runs.
3. Confirm the source filename and run ID above **Candidate evidence**. Synthetic fixtures have an explicit test label. Read the supported-candidate count, recorded facts, and missing context. A result with zero supported candidates is valid; do not describe it as a detected violation.
4. Select **Load before / during / after** on a candidate. Captions show actual sampled timestamps. An unavailable frame stays unavailable.
5. Optionally expand **Optional ChatGPT explanation**. If connected, select **Preview frames and facts for ChatGPT**. Inspect the exact prepared images, facts, and limitations. Only **Send selected frames and facts to ChatGPT** sends that prepared evidence. **Send structured facts only to ChatGPT** is a separate explicit choice without images. If connection or images are unavailable, continue with the recorded evidence. Do not reconnect an existing session just for a demo.
6. Add an optional reviewer note, then choose **Accept candidate** or **Reject candidate** to save it with your disposition. **Reset review** clears that human disposition and note. These actions do not change the machine finding. Optional model observations are also separate and may be wrong.
7. Select **Export review JSON** to download the evidence record and saved human decisions. Reopen the run to verify that the decisions persist. Successful explanations reopen as **Saved explanation** with their original image descriptors and require no new ChatGPT request.

Keep the interpretation narrow: missing detection is not proof of no helmet, stopped traffic is not proof of a red signal, and a still image cannot establish speed. Helmet and wrong-way candidates need appropriate visibility and road context. This demo does not establish real-world accuracy or legal guilt.

Three actual four-second runs on permitted footage produced zero supported candidates. Raw detector boxes included a billboard face mistaken for an unhelmeted head and a car region mistaken for a helmet; association and rule checks did not promote them to supported violations. This is not an accuracy benchmark. Show the recorded evidence and its limits instead of treating an isolated box or a model explanation as a confirmed event.

An additional eight-second, 80-frame run of the same flyover source completed locally in about 45 seconds on the tested CPU runtime. It also produced zero supported candidates. This extends that demonstration; it is not a separate held-out source. The saved review explicitly reports its 200-card display cap and omitted observations.

The subscription image protocol has been verified with clearly labeled synthetic images, including the complete browser preview → explanation → saved-result flow. A live request using real-footage images is still pending specific sharing approval. Real-video detection, local frame preparation, reviewer decisions, and JSON export can be demonstrated without that request.
