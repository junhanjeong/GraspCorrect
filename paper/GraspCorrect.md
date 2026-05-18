# GraspCorrect: Robotic Grasp Correction via Vision-Language Model-Guided Feedback 

Sungjae Lee ${ }^{*}$ Yeonjoo Hong ${ }^{*}$ 1 Kwang In Kim ${ }^{12}$


#### Abstract

Despite significant advancements in robotic manipulation, achieving consistent and stable grasping remains a fundamental challenge, often limiting the successful execution of complex tasks. Our analysis reveals that even state-of-the-art policy models frequently exhibit unstable grasping behaviors, leading to failure cases that create bottlenecks in real-world robotic applications. To address these challenges, we introduce GraspCorrect, a plug-and-play module designed to enhance grasp performance through vision-language model-guided feedback. GraspCorrect employs an iterative visual question-answering framework with two key components: grasp-guided prompting, which incorporates task-specific constraints, and object-aware sampling, which ensures the selection of physically feasible grasp candidates. By iteratively generating intermediate visual goals and translating them into jointlevel actions, GraspCorrect significantly improves grasp stability and consistently enhances task success rates across existing policy models in the RLBench and CALVIN datasets.


## 1. Introduction

Robotic manipulation is a complex, multi-faceted task that requires the seamless integration of environmental perception, action planning, and joint actuation. From grasping delicate wine glasses to performing intricate microsurgery and assembling complex electronics, robots must exhibit precise and adaptive control to interact effectively with diverse objects and environments (Tedrake, 2022).

Recent advances in deep learning have driven significant progress in robotic manipulation, enabling the development

[^0]of diverse and sophisticated robotic policies. For example, R3M uses human video data to pre-train visual representations, providing a cost-effective alternative to collecting extensive robot interaction data (Nair et al., 2022). However, its lack of explicit 3D geometric reasoning limits its precision in spatial manipulation tasks. To address this, PerAct incorporates 3D voxel patches within a transformer architecture to enhance spatial reasoning (Shridhar et al., 2022). Given the computational demands of high-resolution 3D features, more efficient representation techniques have also emerged (Goyal et al., 2023; Gervet et al., 2023). More recently, Ke et al. (2024) proposed integrating diffusion policies with 3D scene representations. The resulting 3D Diffuser Actor enables the direct learning of action distribution spaces conditioned on robot states, achieving state-of-the-art performance in manipulation tasks.

Despite these advancements, the question remains: Has the manipulation problem been fully solved? Our findings suggest otherwise. Even state-of-the-art models like the 3D Diffuser Actor exhibit suboptimal performance in critical tasks. For example, in RLBench, task success rates for scenarios such as block stacking (68.3\%), peg insertion (65.6\%), and shape sorting (44.0\%) remain significantly below the benchmark average of $80 \%$.

At the core of these diverse tasks lies the ability to execute stable, reliable grasps, a fundamental prerequisite for successful manipulation. We hypothesize that unstable grasping is a persistent bottleneck across these tasks. To test this, we conducted a preliminary experiment where stable grasp trajectories were provided up to the grasp point, allowing the policy model to predict actions thereafter. This intervention yielded significant improvements, with task success rates increasing by up to $26.4 \%$ (Figure 1).

To address these challenges, two main strategies have been proposed. One direction focuses on creating extensive datasets of robot demonstrations that encompass a wide range of object properties, grasp types, and environmental conditions. However, such approaches are resource-intensive, requiring substantial computational power for retraining, and are often limited by scope of the datasets (Pumacay et al., 2024). Alternatively, pretrained grasping foundation models like GraspNet (Fang

![](https://cdn.mathpix.com/cropped/f74527f7-3c4f-49fc-9ec5-62a8ddd68762-02.jpg?height=403&width=1600&top_left_y=227&top_left_x=229)
Figure 1. Importance of precise gripper action. Left: Visualization of successful and failed cases in the RLBench insert peg task. Right: Performance improvement on challenging RLBench tasks. By replicating demonstrated stable grasp poses up to the grasping point, we observe substantial improvements in task success rates (\%). This preliminary result highlights the significant impact of robust grasping on the overall performance of end-to-end robotic policy models.

et al., 2020) and Contact-GraspNet (Sundermeyer et al., 2021) generate efficient grasp pose distributions directly from 3D point clouds. While effective in dense point cloud, these models face performance degradation when applied to cost-efficient, sparse 3D representations. Furthermore, their reliance on scenes observed during training makes it difficult to generalize to unseen objects and task categories.

A promising alternative lies in the zero-shot capabilities of Vision-Language Models (VLMs), which excel in scene understanding, reasoning, and generating descriptive text (Yuan et al., 2024). Recent studies demonstrated their effectiveness in robotic manipulation and long-horizon planning tasks (Huang et al., 2024; Kwon et al., 2024). Trained on diverse, large-scale datasets, VLMs offer the adaptability required for robust performance across varying domains.

We hypothesize that integrating VLMs into the grasp correction process can enhance performance by analyzing visual context and suggest adjustments, without requiring extensive retraining or large-scale data collection.

Building on these insights, we propose GraspCorrect, a plug-and-play module specifically designed to enhance grasp reliability in robotic manipulation policies. GraspCorrect operates through a three-stage pipeline. First, it identifies stable grasp positions using VLM guidance that integrates semantic and geometric considerations. Next, it synthesizes a goal-state image via image composition, translating VLM insights into actionable visual objectives. Finally, GoalConditioned Behavioral Cloning (GCBC) converts these visual goals into precise joint-level actions, ensuring accurate execution of the intended grasp.

This architecture-agnostic design allows for seamless integration with existing manipulation policies, enabling grasp refinement without compromising core functionality or requiring extensive retraining. By effectively predicting corrective actions during the critical grasping phase, GraspCorrect significantly improves grasp reliability, achieving
state-of-the-art performance in robotic manipulation.

## 2. Related Work

Policy Learning for Robotic Manipulation: Policy learning enables robots to autonomously develop complex skills for interacting with diverse environments. While Reinforcement Learning (RL) has been widely used to learn optimal policies through interaction with the environment, it often suffers from data inefficiency and requires extensive exploration, limiting its real-world applicability (Tang et al., 2024a). Imitation Learning (IL) leverages expert demonstrations to train policies, reducing the exploration burden. Nonetheless, IL faces challenges with generalization to unseen scenarios (Mandlekar et al., 2020).

Recent advancements in policy learning architectures have gained increasing attention. Transformer-based architectures have demonstrated the ability to capture long-range dependencies and contextual relationships in manipulation tasks (Shridhar et al., 2022; Goyal et al., 2023; 2024). Similarly, diffusion models have been explored to represent complex action distributions (Ke et al., 2024; Black et al., 2024). Despite these developments, robust manipulation remains a significant challenge, particularly in tasks requiring precise grasping (Goyal et al., 2024). This highlights the importance of enhancing grasping capabilities within policy learning to improve overall manipulation.

A closely related approach is SuSIE (Black et al., 2024), which employs an image-editing diffusion model to generate intermediate sub-goal frames for long-horizon manipulation tasks. However, GraspCorrect is motivated by a key insight: correct grasping serves as a crucial milestone for the success of manipulation tasks. As such, GraspCorrect explicitly uses VLMs to assess the likelihood of a correct grasp and to generate corresponding images, a capability that SuSIE lacks. Our empirical observations show that SuSIE struggles to generate appropriate grasp poses, despite being fine-tuned
on a comprehensive dataset that combines both simulation and real-world scenarios (see Figure 4).

Vision Language Models in Robotics: Vision Language Models (VLMs) have significantly advanced robotic interaction through their capabilities in advanced scene understanding, task planning, and descriptive captioning. For example, PaLM-E (Driess et al., 2023) integrates language models with visual and motor capabilities, while RT-1 (Brohan et al., 2022) introduces efficient tokenization techniques for processing high-dimensional camera inputs and motor commands, enabling real-time control.

Despite these advancements, VLMs still struggle to bridge the gap between high-level understanding and low-level control, particularly in maintaining temporal consistency and spatial precision. Even with recent architectures like RT-2 (Brohan et al., 2023), CoPa (Huang et al., 2024), and EmbodiedGPT (Mu et al., 2024), achieving precise manipulation remains difficult. Our work complements these advances by focusing on grasp correction, using VLMs' scene understanding and commonsense reasoning capabilities.

PIVOT (Nasiriany et al., 2024) introduces a visual prompting approach for VLMs, reformulating complex manipulation tasks as iterative visual question-answering (VQA). In each cycle, the image is annotated with visual representations of proposals, allowing the VLM to iteratively refine and select the most suitable option. While we adopt this iterative selection scheme, we observe that it lacks an inherent understanding of physical feasibility and constraints within the visual scene.

As shown in Section 4.1, GraspCorrect effectively addresses these limitations by better constraining sampling strategies and incorporating physical interaction principles.

## 3. Robotic Grasp Correction Framework

We consider robotic manipulation learning, where a policy model learns from demonstration trajectories $\left\{\left(\mathbf{o}_{1}, \mathbf{a}_{1}\right),\left(\mathbf{o}_{2}, \mathbf{a}_{2}\right), \ldots\right\}$ paired with a textual task instruction $l$. Each observation $\mathbf{o}_{t}$ at timestep $t$ consists of RGB-D images, and each action

$$
\mathbf{a}_{t}=\left[\left(\mathbf{a}_{t}^{p}\right)^{\top},\left(\mathbf{a}_{t}^{r}\right)^{\top},\left(\mathbf{a}_{t}^{s}\right)^{\top}\right]^{\top} \in \mathbb{R}^{8}
$$

specifies the end-effector pose through three components: position $\mathbf{a}_{t}^{p} \in \mathbb{R}^{3}$, rotation (quaternion) $\mathbf{a}_{t}^{r} \in \mathbb{R}^{4}$, and binary gripper state $\mathbf{a}_{t}^{s} \in\{0,1\}$.

The policy model aims to learn a mapping that predicts an appropriate action a given the current observation o. GraspCorrect acts as a plug-and-play module for existing policy models, activating at the grasping moment $t(g)$ when the gripper contacts the target object specified in $l$. Using the current grasp pair ( $\mathbf{o}_{t(g)}, \mathbf{a}_{t(g)}$ ) and a pre-grasp observation
$\mathbf{o}_{t(g)-W}$ within a time window $W$ (see Section 3.1), the module predicts a corrective end-effector grasp pose $\mathbf{a}^{o}$ for improved execution.

GraspCorrect operates through three stages. First, in the (VLM-guided) Grasp Detection stage, it identifies stable grasp positions using insights from VLM. In the (Visual) Goal Generation stage, it generates a visual objective in image form. Finally, in the Action Generation stage, it translates the visual goal into precise joint-level actions. Figure 2 illustrates the overall grasp correction process.

### 3.1. VLM-guided Grasp Detection

This stage translates the current observation $\mathbf{o}_{t(g)}$ and task description $l$ into task-oriented contact points $\mathbf{a}_{t(g)}^{p}$ for robotic grippers to ensure stable grasping. Leveraging VLMs, this task can be framed as spatial Visual Question Answering (VQA), which extends traditional VQA tasks (e.g., identifying objects or attributes; "What color is the car?") to include spatial reasoning, such as determining where a robot should grasp an object for a stable lift.

Pre-trained VLMs provide a rich repository of commonsense knowledge for this task. However, their direct application to spatial reasoning presents two main challenges. First, VLMs are optimized for generating textual outputs, making them unsuitable for producing continuous values like coordinates or actions. Second, even state-of-the-art VLMs struggle with complex spatial reasoning (Wang et al., 2024; Chen et al., 2024; Tang et al., 2024b).

To overcome these limitations, we adopt an iterative VQA approach that progressively refines grasp candidates rather than attempting to generate precise spatial coordinates directly. Building on the iterative refinement strategy of PIVOT (Nasiriany et al., 2024), we introduce two key improvements: (1) grasp-guided prompting, which incorporates task-specific constraints, and (2) object-aware sampling, which ensures that generated grasp candidates are physically feasible.

Our approach starts with a top-down 2D observation $\mathbf{o}_{t(g)-W}^{\text {Top }}$, captured $W$ frames before the grasping moment. This earlier frame provides a comprehensive view of object's geometry, as the close-up grasp pose at $t(g)$ may only partially capture the object. The time window size $W$ is fixed at 10 . Using curated prompts tailored to task requirements (see Appendix A.1), the VLM generates a textual description of stable grasp configurations, which serves as a prior for the iterative refinement process.

To ensure precise targeting, we use LangSAM, ${ }^{1}$ a zero-shot text-to-segmentation-mask framework that combines GroundingDINO (Liu et al., 2025) and Segment-

[^1]![](https://cdn.mathpix.com/cropped/f74527f7-3c4f-49fc-9ec5-62a8ddd68762-04.jpg?height=641&width=1705&top_left_y=193&top_left_x=182)
Figure 2. Overview of the GraspCorrect process. This module enhances robotic manipulation by establishing a stable grasp as a critical milestone. In the Grasp Detection stage, task-specific VLM guidance predicts the desired gripper positioning through an iterative question-answering process. The Visual Goal Generation stage then synthesizes a goal-state image via image composition, representing the ideal grasp configuration. Finally, the Action Generation stage predicts and executes corrective actions, improving grasping reliability.

Anything (Kirillov et al., 2023). This segmentation step restricts grasp proposals to the actual object, avoiding hallucinations that could target background elements.

Grasp candidates are initially sampled along the object's contour (Figure 3, circles). The VLM evaluates these points to identify promising candidates (red circles) likely to support stable grasping. New candidates are then generated by sampling from a 1D Gaussian distribution centered around these promising points along the object's contour. The number of iterations $T$ is fixed at 4 , and in the final iteration, a single candidate is selected. A detailed description of this process can be found in Appendix A.1.

### 3.2. Visual Goal Generation

This stage synthesizes a target grasp pose image $\mathbf{o}_{t(g)}^{*}$ that depicts the robotic grippers (left and right), the target object, and their spatial relationships, based on the input observations $\left\{\mathbf{o}_{t(g)}, \mathbf{o}_{t(g)-W}\right\}$, and the grasp points identified by the Grasp Detection stage.

The process starts by restoring the occluded background regions using the LaMa inpainting model (Suvorov et al., 2022) to create a complete background image. The composite image is then constructed by blending the restored background, gripper, and transformed foreground object. Object alignment with the gripper is achieved via conventional image transformations (rotations and translations) guided by the contact point information from the Grasp Detection stage. The resulting goal-state image provides a realistic representation of the desired grasp pose and serves as the foundation for the subsequent Action Generation step.

### 3.3. Action Generation

To achieve low-level joint actuation, we adopt a GoalConditioned Behavior Cloning (GCBC) framework. As a form of imitation learning, behavior cloning trains an agent to replicate expert demonstrations by minimizing the discrepancy between predicted and observed expert actions. Following (Walke et al., 2023), we implement this using Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020), which iteratively refines a Gaussian noise distribution into a data-generating distribution.

Our GCBC policy model $\pi_{\phi}$ comprises a ResNet-34 encoder followed by 3-layer Multi-Layer Perceptron (MLP), parameterized by weights $\phi$. Since observation images are captured from an egocentric top-down perspective, we enhance spatial awareness by incorporating the current action state as a conditioning variable. This facilitates smooth integration of generated output actions into the ongoing trajectory.

The DDPM training loss is formulated as:

$$
\begin{aligned}
\mathcal{L}(\phi)= & \mathbb{E}_{\boldsymbol{\epsilon}, s,\left(\mathbf{a}_{t}, \mathbf{a}_{t}^{*}, \mathbf{o}_{t}, \mathbf{o}_{t}^{*}\right) \sim \mathcal{D}}\left[\lambda\left\|\boldsymbol{\epsilon}^{p}(s)-\boldsymbol{\epsilon}_{\phi}^{p}\left(\mathbf{a}_{t}, \mathbf{o}_{t}, \mathbf{o}_{t}^{*}, s\right)\right\|^{2}\right. \\
& \left.+\left\|\boldsymbol{\epsilon}^{r}(s)-\boldsymbol{\epsilon}_{\phi}^{r}\left(\mathbf{a}_{t}, \mathbf{o}_{t}, \mathbf{o}_{t}^{*}, s\right)\right\|^{2}\right]
\end{aligned}
$$

where $s$ is the diffusion time step, and $t$ represents the grasp time $t(g)$. Here, $\mathbf{o}_{t}$ is the observation image, $\mathbf{a}_{t}$ is the action vector (Equation (1)), $\mathbf{o}_{t}^{*}$ is the goal image, and $\mathbf{a}_{t}^{*}$ is the expert action. The noise vectors $\boldsymbol{\epsilon}_{\phi}^{p}$ and $\boldsymbol{\epsilon}_{\phi}^{r}$, corresponding to position and rotation, respectively, are predicted by $\pi_{\phi}$ to approximate the true noise terms $\boldsymbol{\epsilon}^{p}(s)$ and $\boldsymbol{\epsilon}^{r}(s)$ associated with $\mathbf{a}_{t}^{*}$. The weighting hyperparameter $\lambda$ is set to 0.2 based on validation (see Section 4.1 for details). The expectation in $\mathcal{L}$ is taken over $\mathbf{a}$ and $\boldsymbol{\epsilon}$, where $\mathbf{a}$ comprises $\mathbf{a}^{p}$ and $\mathbf{a}^{r}$,

![](https://cdn.mathpix.com/cropped/f74527f7-3c4f-49fc-9ec5-62a8ddd68762-05.jpg?height=1035&width=819&top_left_y=257&top_left_x=191)
Figure 3. Visualization of iterative grasp point refinement using PIVOT (Nasiriany et al., 2024) (top) and our method (bottom). The circles represent grasp candidates sampled by each algorithm, with red circles indicating those selected for the next sampling stage. Due to its lack of target-specific contextualization, PIVOT often predicts grasp points that fail to make contact with the object. In contrast, our method ensures all selected grasp locations are physically viable. The left and right gripper positions are aligned within the camera's image pane, making it sufficient to generate grasp points near the image boundaries (see Figure 1, left).

and $\boldsymbol{\epsilon}$ includes $\boldsymbol{\epsilon}^{p}$ and $\boldsymbol{\epsilon}^{r}$.
Training data $\mathcal{D}$ is generated within each benchmark environment by systematically perturbing ground-truth grasp poses. Further details on the policy model and data generation process can be found in Appendix A.2.

### 3.4. Discussion

Complementary Roles of VLMs and Behavioral Control: Our approach combines a VLM for grasp detection with GCBC for action generation, recognizing the limitations of VLMs in directly synthesizing precise grasping actions. While VLMs excel in scene understanding and high-level planning, they struggle with the fine-grained control required for embodied manipulation.

![](https://cdn.mathpix.com/cropped/f74527f7-3c4f-49fc-9ec5-62a8ddd68762-05.jpg?height=675&width=826&top_left_y=264&top_left_x=1061)
Figure 4. Evaluation of diffusion-based models for generating goalstate images in robotic manipulation tasks. The input image (topleft) shows the initial gripper configuration approaching a blue square object, while the expected output (bottom-left) represents the ground-truth stable grasp pose from a successful RLBench insert peg demonstration. Existing models struggle to accurately capture the required details, spatial arrangements, and contextual relevance essential for precise robotic grasping.

In our preliminary experiments, directly using VLMs for action prediction $\left(\mathbf{a} \in \mathbb{R}^{8}\right)$ based on current observations, actions, and task descriptions often resulted in unrealistic and physically implausible outputs. This validates our decision to partition the manipulation pipeline, using VLMs for perception and planning while relying on a specialized GCBC module for precise control.

Advantages of Image-Based Intermediate Representations: GraspCorrect uses images as the intermediate goal representation. This decision is grounded in several key advantages. First, visual representations capture rich spatial and contextual information that might be lost or ambiguous in text-based descriptions. Images naturally encode crucial manipulation-relevant features such as spatial relationships, object orientations, and grasp configurations in a concrete, unambiguous manner.

Second, VLMs have been extensively trained on large-scale visual data, making them particularly adept at processing and reasoning about image-based information. This alignment enables our system to fully exploit VLMs' advanced visual understanding and reasoning capabilities while maintaining a clear and interpretable interface for high-level decision-making.

Third, recent successes in robotic manipulation using synthesized goal images, such as SuSIE (Black et al., 2024) and GR-MG (Li et al., 2025), further demonstrate the effectiveness of this approach.

However, image-based representations also present certain limitations, particularly in handling occlusions and capturing dynamic physical properties. Future work could explore incorporating additional modalities, such as 3D point clouds or force feedback, to provide a richer and more comprehensive representation of the goal state.

Alternative Visual Goal Generation Strategies: The expected goal-state grasp pose typically requires minor adjustments from the current grasp pose provided by a pre-trained policy model. While path-planning algorithms like Rapidlyexploring Random Trees (RRT) (Steven, 1998) might seem applicable, they are unsuitable for this context as they require exact destination coordinates, which are challenging to derive due to mismatches between the egocentric robotic camera and the robot's coordinate system. Consequently, pinpointing the precise coordinate indicated by VLMs' responses becomes challenging.

An alternative approach involves using image-editing or image-generation diffusion models prompted by the VLMgenerated grasp descriptions. We tested four such models in RLBench: DALL-E (image generation) (Ramesh et al., 2021), SuSIE (Black et al., 2024), DiffEdit (Couairon et al., 2023), and Imagic (image editing) (Kawar et al., 2023). Following SuSIE's approach, we employed a fine-tuned version of InstructPix2Pix (Brooks et al., 2023) specifically adapted for manipulation tasks.

As shown in Figure 4, these models often fail to generate accurate and reliable results. For instance, DALL-E tends to generate overly intricate robotic grippers that deviate significantly from the actual gripper design. DiffEdit and SuSIE misrepresent the square's orientation, failing to align it with the expected grasp pose, while Imagic introduces unrealistic human fingers.

In contrast, a simple image blending technique proves highly effective for generating accurate and realistic composite images of the goal state. This approach preserves the structural integrity of the target object and maintains the spatial relationships critical for successful manipulation.

## 4. Experiments

To assess the effectiveness of our GraspCorrect module, we conducted experiments on RLBench (James et al., 2020) and CALVIN (Mees et al., 2022). For all baseline models, we used the pre-trained weights provided by the respective authors. VLM guidance was provided by ChatGPT-4o (OpenAI, 2025) in its standard form without modifications. All
experiments followed the default setup of each benchmark, using the Franka Panda Arm with a parallel-jaw gripper (see Figure 2). While our evaluation focuses on this specific configuration, the framework is adaptable to other VLMs and vision-based robotic manipulation tasks.

RLBench: RLBench is a widely used benchmark in robotic learning research (Shridhar et al., 2022; Gervet et al., 2023; Ke et al., 2024; Goyal et al., 2024), providing 100 diverse manipulation tasks that simulate real-world scenarios. We focus on 18 fundamental manipulation tasks that are broadly recognized within the robotics community, following the evaluation framework proposed by (Shridhar et al., 2022). Among these, the first five tasks listed in Table 1, in raster order, represent particularly challenging scenarios where state-of-the-art methods consistently achieve success rates below $80 \%$. For baseline comparisons, we evaluated four manipulation models: PerAct (Shridhar et al., 2022), Act3D (Gervet et al., 2023), 3D Diffuser Actor (Ke et al., 2024) and RVT-2 (Goyal et al., 2024).

Table 1 summarizes the results. While 3D Diffuser Actor and RVT-2 demonstrate strong baseline performance, they occasionally struggle during critical grasping moments. Even minor inaccuracies in grasp execution can lead to task failures, despite the preceding execution sequence being correct (see task failure example in Figure 1).

Integrating GraspCorrect with these architectures led to substantial performance improvements, averaging 18.3\% and $5.5 \%$, respectively, across evaluation tasks. By providing targeted refinement during the grasping phase, GraspCorrect effectively mitigates these models' primary performance bottleneck while preserving their advanced trajectory generation and dynamic control capabilities. This highlights how precise intervention at key manipulation stages can further enhance reliability, even in already robust policies.

For PerAct and Act3D, performance gains were consistently observed, but they were more modest. We attribute this to two key limitations. First, these models struggle with precise height prediction for keypoints, often resulting in complete grasp failures before GraspCorrect can intervene. Since our module activates specifically during the grasping phase, it cannot compensate for cases where the initial approach fails to position the gripper within a viable correction range.

Second, these models exhibit multi-task performance limitations that extend beyond grasping. While GraspCorrect improves initial grasp stability, failures still occur in subsequent manipulation stages, particularly in object release orientation and adherence to language-conditioned constraints.

Overall, GraspCorrect often significantly enhances task performance, with particularly strong improvements in the most challenging tasks (the first five in Table 1). This supports

Table 1. Performance across different manipulation frameworks on 18 tasks from the RLBench dataset. For each example in each task, the initial position (and orientation) of the target object are randomly generated three times. The mean task success rate (\%) $\pm$ standard deviations is computed across all examples and random object placement combinations within each task. The dash symbol (-) indicates that GraspCorrect remains inactive for tasks that do not involve object grasping.
|  | stack blocks | sort shape | insert peg | stack cups | place cups | sweep to dustpan | turn tap | put in drawer | close jar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PerAct + Ours | $29.3 \pm 6.1 33.3 \pm 6.1$ | $17.3 \pm 2.3 21.3 \pm 2.3$ | $5.3 \pm 2.3 8.0 \pm 6.9$ | $0.0 \pm 0.0 0.0 \pm 0.0$ | $1.3 \pm 2.3 1.3 \pm 2.3$ | $42.7 \pm 8.3 42.7 \pm 8.3$ | $82.7 \pm 4.6 84.0 \pm 4.0$ | $53.3 \pm 2.3 64.0 \pm 6.9$ | $50.7 \pm 2.3 74.7 \pm 2.3$ |
| Act3D + Ours | $4.0 \pm 4.0 6.7 \pm 2.3$ | $36.0 \pm 4.0 37.3 \pm 6.1$ | $16.0 \pm 0.0 24.0 \pm 4.0$ | $6.7 \pm 2.3 6.7 \pm 2.3$ | $9.3 \pm 6.1 9.3 \pm 6.1$ | $88.0 \pm 10.6 88.0 \pm 10.6$ | $90.7 \pm 6.1 94.7 \pm 4.6$ | $93.3 \pm 4.6 96.0 \pm 2.3$ | $96.0 \pm 0.0 98.7 \pm 2.3$ |
| 3D Diff. Act. + Ours | $66.7 \pm 4.5 76.0 \pm 4.0$ | $46.0 \pm 4.0 60.0 \pm 4.0$ | $65.3 \pm 2.3 82.7 \pm 2.0$ | $40.0 \pm 4.0 65.3 \pm 6.1$ | $21.3 \pm 6.1 45.3 \pm 16.2$ | $89.3 \pm 1.5 89.3 \pm 1.5$ | $97.3 \pm 2.3 98.7 \pm 2.3$ | $90.7 \pm 2.3 94.0 \pm 2.8$ | $98.7 \pm 2.3 100.0 \pm 0.0$ |
| RVT-2 + Ours | $78.7 \pm 4.6 82.7 \pm 4.6$ | $34.7 \pm 9.2 49.3 \pm 6.1$ | $48.0 \pm 4.0 54.7 \pm 4.6$ | $73.3 \pm 8.3 82.7 \pm 2.3$ | $44.0 \pm 4.0 61.3 \pm 6.1$ | $100.0 \pm 0.0 100.0 \pm 0.0$ | $98.7 \pm 2.3 98.7 \pm 2.3$ | $97.3 \pm 2.3 100.0 \pm 0.0$ | $100.0 \pm 0.0 100.0 \pm 0.0$ |
|  | screw bulb | place wine | meat off grill | put in cupboard | open drawer | drag stick | put in safe | push buttons | slide block |
| PerAct + Ours | $25.3 \pm 2.3 25.3 \pm 2.3$ | $48.0 \pm 6.9 49.3 \pm 6.1$ | $70.7 \pm 2.3 80.0 \pm 0.0$ | $33.3 \pm 8.3 36.0 \pm 8.0$ | $50.7 \pm 2.3 96.0 \pm 0.0$ | $89.3 \pm 6.1 90.7 \pm 6.1$ | $80.0 \pm 6.9 80.0 \pm 6.9$ | $94.7 \pm 2.3$ - | $80.0 \pm 13.9$ - |
| Act3D + Ours | $33.3 \pm 8.3 45.3 \pm 2.3$ | $60.0 \pm 8.0 60.0 \pm 8.0$ | $97.3 \pm 2.3 97.3 \pm 2.3$ | $66.7 \pm 4.6 69.3 \pm 2.3$ | $85.3 \pm 6.1 93.3 \pm 8.3$ | $66.7 \pm 2.3 68.0 \pm 0.0$ | $98.7 \pm 2.3 98.7 \pm 2.3$ | $93.3 \pm 2.3$ | $93.3 \pm 4.6$ |
| 3D Diff. Act. + Ours | $69.3 \pm 2.3 86.7 \pm 2.3$ | $88.0 \pm 8.0 93.3 \pm 2.3$ | $90.7 \pm 6.1 97.3 \pm 4.6$ | $78.7 \pm 2.3 90.7 \pm 2.3$ | $90.7 \pm 4.6 98.7 \pm 2.3$ | $98.7 \pm 2.3 98.7 \pm 2.3$ | $97.3 \pm 2.3 97.3 \pm 2.3$ | $97.3 \pm 2.3$ | $98.7 \pm 2.3$ |
| RVT-2 + Ours | $92.0 \pm 0.0 92.0 \pm 0.0$ | $89.3 \pm 4.6 94.7 \pm 2.3$ | $97.3 \pm 2.3 98.7 \pm 2.3$ | $68.0 \pm 4.0 72.0 \pm 0.0$ | $72.0 \pm 6.9 88.0 \pm 4.0$ | $100.0 \pm 0.0 100.0 \pm 0.0$ | $97.3 \pm 4.6 97.3 \pm 4.6$ | $100.0 \pm 0.0$ | $93.3 \pm 4.6$ |


our hypothesis that unstable grasping remains a major bottleneck in robotic manipulation. Figure 6 provides a comparative visualization of manipulation trajectories, illustrating the impact of GraspCorrect on task completion.

Performance gains are more moderate in certain cases (last four columns in Table 1). In these tasks, baseline models already exhibited near-perfect performance, or the tasks did not involve object grasping, meaning our module was not invoked. Importantly, GraspCorrect did not degrade performance in any case.

The results also highlight areas for improvement. GraspCorrect's impact is limited in cases where failures occur earlier in the manipulation pipeline, such as approach planning or object localization errors. Additionally, since our module does not fully modify the action policy, it does not address post-grasp actions, including object placement orientation or adherence to task constraints. Future work could extend GraspCorrect to intervene at earlier manipulation stages or integrate it within a broader policy refinement framework to enhance full-sequence task execution.

Figure 7 shows representative examples across four manipulation scenarios. GraspCorrect consistently generates grasp poses closely aligned with ground-truth demonstrations. In contrast, Contact-GraspNet struggles to generalize to unseen
object-task combinations beyond its training settings.

CALVIN: The CALVIN dataset provides an environment for learning long-horizon language-conditioned tasks, encompassing 34 distinct tasks, such as opening drawers and pushing blocks (Mees et al., 2022). Following (Mees et al., 2022), we adopted the $\mathrm{ABC} \rightarrow \mathrm{D}$ configuration, where models are trained in three different environments (A, B, and C) and evaluated in an unseen environment (D).

Each scenario consists of five consecutive tasks, evaluated using sequential success rates and the average length metric. The former averages success rate across tasks, where failure in one task leads to failure in subsequent tasks. The latter is the sum of success rates over the five tasks, ranging from 0 to 5 . We report the mean score for 100 scenarios. For baselines, we evaluated four models: GR-MG (Li et al., 2025), MoDE (Reuss et al., 2024), 3D Diffuser Actor (Ke et al., 2024) and SuSIE (Black et al., 2024).

Table 2 shows the results. Consistent with the findings from RLBench, integrating GraspCorrect with baseline models improved performance across evaluation tasks, with average length enhancements ranging from 0.1 to 0.8 . The gains for GR-MG and MoDE were relatively modest, which can be attributed to their already high proficiency in execut-
ing benchmark tests (see Table 2). In particular, GR-MG and MoDE achieved near-perfect accuracies of 0.99 and 1.00 , respectively, when only the first task was considered. Nevertheless, the results indicate that incorporating grasp pose guidance ultimately enhances execution performance in long-horizon tasks. Importantly, GraspCorrect never degraded the performance in any scenario.

Table 2. Performance on the CALVIN benchmark: Average success rates for each consecutive task and the corresponding average length for each baseline model.
|  | \# of consecutive tasks completed |  |  |  |  | Avg. len |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | 1 | 2 | 3 | 4 | 5 |  |
| SuSIE | 0.90 | 0.72 | 0.48 | 0.35 | 0.25 | $2.7 \pm 1.7$ |
| + Ours | 0.96 | 0.86 | 0.72 | 0.56 | 0.41 | $3.5 \pm 1.6$ |
| 3D Diff. Act. | 0.93 | 0.79 | 0.65 | 0.55 | 0.43 | $3.4 \pm 1.8$ |
| + Ours | 0.97 | 0.87 | 0.77 | 0.69 | 0.54 | $3.9 \pm 1.5$ |
| GR-MG | 0.99 | 0.94 | 0.85 | 0.78 | 0.67 | $4.2 \pm 1.3$ |
| + Ours | 0.99 | 0.95 | 0.87 | 0.82 | 0.74 | $4.4 \pm 1.3$ |
| MoDE | 1.00 | 0.94 | 0.88 | 0.81 | 0.72 | $4.4 \pm 1.2$ |
| + Ours | 1.00 | 0.96 | 0.89 | 0.84 | 0.77 | $4.5 \pm 1.1$ |


### 4.1. Ablation Study

A Comparative Evaluation of PIVOT and GraspCorrect: PIVOT streamlines robotic control by directly generating continuous actions from VLM outputs (Nasiriany et al., 2024). It relies on a spatial mapping mechanism that uses camera matrices to project 3D locations onto the image plane, where grasp candidates are sampled (Figure 3). Unlike GraspCorrect, PIVOT bypasses intermediate goal image synthesis and explicit action generation.

While PIVOT's direct approach may seem more efficient, we observed that GraspCorrect's additional processing stages significantly enhances manipulation reliability (Figure 3). The key difference lies in GraspCorrect's focus on grasp stability and action space exploration: PIVOT's full 3D sampling struggles with the vast search space, often missing optimal grasp points. GraspCorrect mitigates this by leveraging egocentric views, combined with grasp-guided prompting and object-aware sampling to effectively communicate task-specific and physical constraints to the VLMs.

Our ablation studies demonstrate the impact of these core components for VLM-guided grasp detection across top five challenging tasks in RLBench (Table 3). Grasp-guided prompting improved the task success rate from $30.61 \%$ to $42.52 \%$, while object-constrained sampling further boosted it to $73.81 \%$, demonstrating their combined effectiveness in achieving robust, efficient grasp detection.

Unlike full manipulation task execution, quantitatively ana-

Table 3. Ablation study results demonstrating the impact of graspguided prompting and object-aware sampling on task success rate.
| Configuration |  | Success rate |
| :---: | :---: | :---: |
| Grasp-guided prompt. | Object-aware sampl. |  |
| $x$ | $x$ | 30.6 |
| $\checkmark$ | $x$ | 42.5 |
| $\checkmark$ | $\checkmark$ | 73.8 |


Table 4. Effect of action component weighting $\lambda$ (insert peg task; RLBench).
| $\lambda$ | 1.0 | 0.5 | 0.2 | 0.1 |
| :---: | :---: | :---: | :---: | :---: |
| Success rate | $78.7 \pm 16.2$ | $77.3 \pm 9.2$ | $82.7 \pm 2.3$ | $77.3 \pm 10.1$ |


lyzing grasp quality is challenging due to the lack of exact ground-truth data. Instead, we evaluated grasp accuracy based on the deviation of the action vector generated by GraspCorrect $\mathbf{a}^{o}$ (Equation (1)) from those in successful RLBench demonstrations, treating them as (semi-)ground truth. GraspCorrect achieved an average squared Euclidean distance of 1.15, whereas it showed 1.88 before performing GraspCorrect, indicating a substantial improvement in grasp estimation (insert peg task).

Furthermore, when considering a prediction correct if its squared distance from the ground truth is below 1.52, GraspCorrect demonstrated an average improvement of $18.75 \%$ in accurate grasp generation.

Impact of Action Component Weighting $\lambda$ : The weighting hyperparameter $\lambda$ in $\mathcal{L}$ was set to 0.2 based on RLBench validation, emphasizing the importance of orientation in effective grasping (see Table 4). As $\lambda$ deviates from this optimum value, performance degrades gracefully (Table 4).

## 5. Conclusion

This work introduced GraspCorrect, a plug-and-play module designed to enhance existing robotic manipulation policies by providing precise grasping guidance. By integrating high-level semantic insights from vision-language models with detailed low-level action refinement through goalconditioned behavioral cloning and visual goal generation, GraspCorrect significantly advances robotic manipulation capabilities. Its architecture-agnostic design enables efficient grasp refinement without requiring extensive retraining, as demonstrated across diverse manipulation tasks in RLBench and CALVIN experiments.

Limitations and Future Work: One limitation of our approach is its reliance on top-view imagery, which simplifies
grasp guidance but may overlook crucial geometric features in complex 3D environments. Additionally, the framework is less effective in addressing failures that occur earlier in the manipulation pipeline or during post-grasping phases, as it primarily focuses on refining the grasp itself. Beyond these limitations, future work could explore extending the framework to handle dynamic scenes and deformable objects. Our approach relies solely on visual inputs; incorporating force and tactile feedback could further refine grasp correction allowing the system to adjust its grip based on material properties, surface friction, and object stability.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant (No. 2021R1A2C2012195, 33\%) and the Institute of Information \& Communications Technology Planning \& Evaluation (IITP) grants (No. RS-2019-II191906, AI Graduate School Program, POSTECH, $33 \%$; and No. RS-2022-II220290, Visual Intelligence for Space-Time Understanding and Generation, 33\%), funded by the Korean government (MSIT).

## References

Black, K., Nakamoto, M., Atreya, P., Walke, H., Finn, C., Kumar, A., and Levine, S. Zero-shot robotic manipulation with pretrained image-editing diffusion models. In ICLR, 2024.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jackson, T., Jesmonth, S., Joshi, N. J., Julian, R., Kalashnikov, D., Kuang, Y., Leal, I., Lee, K.-H., Levine, S., Lu, Y., Malla, U., Manjunath, D., Mordatch, I., Nachum, O., Parada, C., Peralta, J., Perez, E., Pertsch, K., Quiambao, J., Rao, K., Ryoo, M., Salazar, G., Sanketi, P., Sayed, K., Singh, J., Sontakke, S., Stone, A., Tan, C., Tran, H., Vanhoucke, V., Vega, S., Vuong, Q., Xia, F., Xiao, T., Xu, P., Xu, S., Yu, T., and Zitkovich, B. RT-1: Robotics transformer for real-world control at scale. arXiv:2212.06817, 2022.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., Florence, P., Fu, C., Arenas, M. G., Gopalakrishnan, K., Han, K., Hausman, K., Herzog, A., Hsu, J., Ichter, B., Irpan, A., Joshi, N., Julian, R., Kalashnikov, D., Kuang, Y., Leal, I., Lee, L., Lee, T.-W. E., Levine, S., Lu, Y., Michalewski, H., Mordatch, I., Pertsch, K., Rao, K., Reymann, K., Ryoo, M., Salazar, G., Sanketi, P., Sermanet, P., Singh, J., Singh, A., Soricut, R., Tran, H., Vanhoucke, V., Vuong, Q., Wahid, A., Welker, S., Wohlhart, P., Wu, J., Xia, F., Xiao, T., Xu, P., Xu, S., Yu, T., and Zitkovich,
B. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv:2307.15818, 2023.

Brooks, T., Holynski, A., and Efros, A. A. InstructPix2Pix: Learning to follow image editing instructions. In CVPR, 2023.

Chen, B., Xu, Z., Kirmani, S., Ichter, B., Driess, D., Florence, P., Sadigh, D., Guibas, L., and Xia, F. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024.

Couairon, G., Verbeek, J., Schwenk, H., and Cord, M. DiffEdit: Diffusion-based semantic image editing with mask guidance. In ICLR, 2023.

Driess, D., Xia, F., Sajjadi, M. S. M., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., Huang, W., Chebotar, Y., Sermanet, P., Duckworth, D., Levine, S., Vanhoucke, V., Hausman, K., Toussaint, M., Greff, K., Zeng, A., Mordatch, I., and Florence, P. PaLM-E: An embodied multimodal language model. arXiv:2303.03378, 2023.

Fang, H., Wang, C., Gou, M., and Lu, C. GraspNet-1Billion: A large-scale benchmark for general object grasping. In CVPR, 2020.

Gervet, T., Xian, Z., Gkanatsios, N., and Fragkiadaki, K. Act3D: Infinite resolution action detection transformer for robotic manipulation. In CoRL, 2023.

Goyal, A., Xu, J., Guo, Y., Blukis, V., Chao, Y.-W., and Fox, D. RVT: Robotic view transformer for 3D object manipulation. In CoRL, 2023.

Goyal, A., Blukis, V., Xu, J., Guo, Y., Chao, Y.-W., and Fox, D. RVT-2: Learning precise manipulation from few demonstrations. arXiv:2406.08545, 2024.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In CVPR, 2016.

Hendrycks, D. and Gimpel, K. Gaussian error linear units (GELUs). arXiv:1606.08415, 2016.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. NeurIPS, 2020.

Huang, H., Lin, F., Hu, Y., Wang, S., and Gao, Y. CoPa: General robotic manipulation through spatial constraints of parts with foundation models. arXiv:2403.08248, 2024.

James, S., Ma, Z., Arrojo, D. R., and Davison, A. J. RLBench: The robot learning benchmark \& learning environment. IEEE RA-L, 2020.

Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., and Irani, M. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023.

Ke, T.-W., Gkanatsios, N., and Fragkiadaki, K. 3D Diffuser Actor: Policy diffusion with 3D scene representations. In CoRL, 2024.

Kingma, D. P. and Ba, J. L. Adam: A method for stochastic optimization. arXiv:1412.6980, 2014.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., and Girshick, R. Segment anything. In CVPR, 2023.

Kwon, T., Palo, N. D., and Johns, E. Language models as zero-shot trajectory generators. IEEE RA-L, 2024.

Li, P., Wu, H., Huang, Y., Cheang, C., Wang, L., and Kong, T. GR-MG: Leveraging partially-annotated data via multimodal goal-conditioned policy. IEEE RA-L, 2025.

Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., and Zhang, L. Grounding DINO: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2025.

Mandlekar, A., Xu, D., Martín-Martín, R., Savarese, S., and Fei-Fei, L. Learning to generalize across long-horizon tasks from human demonstrations. arXiv:2003.06085, 2020.

Mees, O., Hermann, L., Rosete-Beas, E., and Burgard, W. CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE RA-L, 2022.

Mu, Y., Zhang, Q., Hu, M., Wang, W., Ding, M., Jin, J., Wang, B., Dai, J., Qiao, Y., and Luo, P. EmbodiedGPT: Vision-language pre-training via embodied chain of thought. NeurIPS, 2024.

Nair, S., Rajeswaran, A., Kumar, V., Finn, C., and Gupta, A. R3M: A universal visual representation for robot manipulation. arXiv:2203.12601, 2022.

Nasiriany, S., Xia, F., Yu, W., Xiao, T., Liang, J., Dasgupta, I., Xie, A., Driess, D., Wahid, A., Xu, Z., Vuong, Q., Zhang, T., Lee, T.-W. E., Lee, K.-H., Xu, P., Kirmani, S., Zhu, Y., Zeng, A., Hausman, K., Heess, N., Finn, C., Levine, S., and Ichter, B. PIVOT: Iterative visual prompting elicits actionable knowledge for VLMs. arXiv:2402.07872, 2024.

OpenAI. ChatGPT (jan 14 version). https://chat. openai.com/, 2025. Accessed: 2025-01-14.

Pumacay, W., Singh, I., Duan, J., Krishna, R., Thomason, J., and Fox, D. THE COLOSSEUM: A benchmark for evaluating generalization for robotic manipulation. arXiv:2402.08191, 2024.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. In ICML, 2021.

Reuss, M., Pari, J., Agrawal, P., and Lioutikov, R. Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning. arXiv:2412.12953, 2024.

Shridhar, M., Manuelli, L., and Fox, D. Perceiver-Actor: A multi-task transformer for robotic manipulation. In CoRL, 2022.

Steven, L. Rapidly-exploring random trees: A new tool for path planning. Technical report, Department of Computer Science, Iowa State University, 1998.

Sundermeyer, M., Mousavian, A., Triebel, R., and Fox, D. Contact-GraspNet: Efficient 6-DoF grasp generation in cluttered scenes. ICRA, 2021.

Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., and Lempitsky, V. Resolution-robust large mask inpainting with fourier convolutions. In WACV, 2022.

Tang, C., Abbatematteo, B., Hu, J., Chandra, R., Martín, R. M., and Stone, P. Deep reinforcement learning for robotics: A survey of real-world successes. Annual Review of Control, Robotics, and Autonomous Systems, 8, 2024a.

Tang, Y., Qu, A., Wang, Z., Zhuang, D., Wu, Z., Ma, W., Wang, S., Zheng, Y., Zhao, Z., and Zhao, J. Sparkle: Mastering basic spatial capabilities in vision language models elicits generalization to composite spatial reasoning. arXiv:2410.16162, 2024b.

Tedrake, R. Robotic Manipulation: Perception, Planning, and Control. 2022. URL http://manipulation. mit.edu.

Walke, H., Black, K., Lee, A., Kim, M. J., Du, M., Zheng, C., Zhao, T., Hansen-Estruch, P., Vuong, Q., He, A., Myers, V., Fang, K., Finn, C., and Levine, S. BridgeData V2: A dataset for robot learning at scale. In CoRL, 2023.

Wang, J., Ming, Y., Shi, Z., Vineet, V., Wang, X., Li, Y., and Joshi, N. Is a picture worth a thousand words? Delving into spatial reasoning for vision language models. arXiv:2406.14852, 2024.

Wu, Y. and He, K. Group normalization. In ECCV, 2018.
Yuan, Y., Li, W., Liu, J., Tang, D., Luo, X., Qin, C., Zhang, L., and Zhu, J. Osprey: Pixel understanding with visual instruction tuning. In CVPR, 2024.

## A. Implementation and Experimental Details

## A.1. VLM-guided Grasp Detection

## Grasp-Guided Prompts:

```
You are a robot equipped with a parallel-jaw gripper, performing the task '{task_desc}'.
Analyze the provided pre-grasp pose of an object and specify precise contact positions
for each robot gripper to achieve a stable grasp. Describe the contact position as
much detail as possible using numerical expressions. Avoid using exact coordinates.
Respond in the format: 'Left: [1 sentence starting with ''Position the left
gripper'']. Right: [1 sentence starting with ''Position the right gripper'']'. Let's
think step by step.
```

Every RLBench task has various textual descriptions task_desc (e.g., "put the ring on the maroon spoke", "open the bottom drawer", "place 1 cup on the cup holder") to be used for linguistic learning in robotic tasks.

## Iterative VQA:

```
INSTRUCTIONS: You are tasked to locate an object, region, or point in space in the
given annotated image according to a description. The image is annotated with numbered
circles.
Choose the top {top_n} circles that have the most overlap with and/or is closest to what
the description is describing in the image. You are a five-time world champion in this
game. Give a one sentence analysis of why you chose those points. Provide your answer
at the end in a valid JSON of this format: "points": [].
DESCRIPTION: {description}
IMAGE: {image}
```

Grasp candidates are chosen by iteratively querying VLM using the above prompt adapted from PIVOT (Nasiriany et al., 2024)'s GitHub repository. In this process, we set top_n value to 3 , while the description corresponds to the output generated during the grasp-guided prompting phase.

Our module introduces a sequential approach to parallel-jaw gripper positioning that significantly improves spatial coherence during the VQA process. This method begins by determining the left gripper's positioning, marking it on the input image, and then incorporating this information through additional language prompts ("Be aware that the red circle indicates the left gripper's contact position") when determining the right gripper's position. We find that independent positioning which determines contact points for both grippers in isolation frequently results in suboptimal or physically infeasible grasp configurations, primarily due to the lack of inter-gripper spatial awareness. By explicitly incorporating the first gripper's position into the decision process for the second gripper, our sequential method enables the VLM to maintain comprehensive spatial awareness throughout the grasp correction process.

## A.2. Action Generation

Model Architecture: The policy network architecture consists of a ResNet-34 (He et al., 2016) with Group Normalization (Wu \& He, 2018), which processes the current and goal observation images stacked along the channel dimension. The encoder's output is then passed through a 3-layer Multi-Layer Perceptron (MLP) with 256 hidden units and Swish activations (Hendrycks \& Gimpel, 2016) in each layer. This MLP outputs the mean and standard deviation for a Gaussian action distribution.

Training is performed using the Adam optimizer (Kingma \& Ba, 2014) with a learning rate of 5e-4, a linear warmup schedule over 2,000 steps, and a batch size of 256. Prior to concatenation, both current and goal images undergo standard image augmentations, including random cropping, as well as brightness, contrast, saturation, and hue adjustments.

Data Generation Protocol: Our action generation policy requires paired data consisting of observation-action tuples ( $\mathbf{o}_{t(g)}$, $\mathbf{a}_{t(g)}, \mathbf{o}_{t(g)}^{*}, \mathbf{a}_{t(g)}^{*}$ ), where states requiring grasp correction are paired with their corresponding stable grasp configurations. We implement a two-stage data collection protocol within the RLBench environment to generate these training pairs.

For collecting correction-needed states $\left(\mathbf{o}_{t(g)}, \mathbf{a}_{t(g)}\right)$, we first initialize a simulation environment comprising a tabletop workspace, target object, and Franka Panda robot. The motion path is defined by waypoints (Figure 5 left) that serve

![](https://cdn.mathpix.com/cropped/f74527f7-3c4f-49fc-9ec5-62a8ddd68762-12.jpg?height=448&width=1367&top_left_y=225&top_left_x=348)
Figure 5. Data generation process for policy training. Left: waypoints are randomly varied to introduce realistic grasping variations. Middle: the resulting grasp pose with variation is saved. Right: the grasp pose without waypoint variation represents the stable grasp.

as reference points for RLBench's path planning algorithm. By introducing controlled randomization to these waypoint positions and orientations, we generate realistic variations in grasp attempts. At the moment of grasping, we record both the observation and the executed action vector. To obtain stable reference states $\left(\mathbf{o}_{t(g)}^{*}, \mathbf{a}_{t(g)}^{*}\right)$, we repeat the grasping sequence under identical conditions but without waypoint randomization. Through this systematic process, we generate 200 paired examples for each manipulation task, providing a balanced dataset for policy learning.

## B. Visualization

![](https://cdn.mathpix.com/cropped/f74527f7-3c4f-49fc-9ec5-62a8ddd68762-13.jpg?height=1275&width=1292&top_left_y=335&top_left_x=386)
Figure 6. Visualization of manipulation processes with and without GraspCorrect. (Left) insert peg and (Right) place cups. The GraspCorrect-assisted manipulation process consists of four stages: initial grasp pose, visual goal generation, actual grasp execution, and final action. By effectively correcting initial grasp inaccuracies, GraspCorrect ensures successful task completion.

![](https://cdn.mathpix.com/cropped/f74527f7-3c4f-49fc-9ec5-62a8ddd68762-14.jpg?height=1230&width=1634&top_left_y=672&top_left_x=214)
Figure 7. Comparison of grasp poses across four manipulation tasks: insert peg, sort shape (star), put in cupboard (mustard), and place cups; Top: ground-truth grasps from successful manipulations in RLBench. Middle: Contact-GraspNet (red markers indicate the predicted gripper poses). Bottom: ours.


[^0]:    *Equal contribution ${ }^{1}$ Graduate School of AI, POSTECH ${ }^{2}$ Dept. of EE, POSTECH. Correspondence to: Sungjae Lee [leeeesj@postech.ac.kr](mailto:leeeesj@postech.ac.kr), Yeonjoo Hong [yeonjooh@postech.ac.kr](mailto:yeonjooh@postech.ac.kr), Kwang In Kim [kimkin@postech.ac.kr](mailto:kimkin@postech.ac.kr).

[^1]:    ${ }^{1}$ https://github.com/luca-medeiros/lang-segment-anything

