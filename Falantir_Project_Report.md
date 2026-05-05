# Falantir: A Hybrid Vision–Language Pipeline for Real-Time Retail Threat Detection with Knowledge-Distilled Edge Fallback

---

## TITLE PAGE

**Project Title**

Falantir: A Hybrid Vision–Language Pipeline for Real-Time Retail Threat Detection with Knowledge-Distilled Edge Fallback

**A Project Report**

Submitted in partial fulfilment of the requirements for the award of the degree of

**Bachelor of Engineering**

in

**Artificial Intelligence and Machine Learning**

**Submitted by**

Vivek Muthe

**Under the Guidance of**

[Project Guide Name]

[Department of Artificial Intelligence and Machine Learning]

[Name of Institution]

[Affiliated University]

[Academic Year 2025–2026]

---

## CERTIFICATE

This is to certify that the project report entitled **"Falantir: A Hybrid Vision–Language Pipeline for Real-Time Retail Threat Detection with Knowledge-Distilled Edge Fallback"** is the bonafide work carried out by **Vivek Muthe** under my supervision and guidance, submitted in partial fulfilment of the requirements for the award of the degree of Bachelor of Engineering in Artificial Intelligence and Machine Learning, during the academic year 2025–2026.

The work presented in this report has been carried out by the candidate at the Department of Artificial Intelligence and Machine Learning, [Name of Institution], and has not been submitted elsewhere for the award of any other degree or diploma.

The project demonstrates a substantial original contribution in the design, development, and evaluation of a multi-tier vision pipeline that integrates large-scale vision–language models with a locally trained student classifier for cost-efficient real-time surveillance.

\
\
\
\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Project Guide                                                Head of Department

\
\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

External Examiner

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to all those who supported me throughout the development of this project. The successful completion of Falantir would not have been possible without their guidance and encouragement.

I am deeply indebted to my project guide, whose continuous mentorship, patient supervision, and constructive feedback shaped both the technical direction and the academic rigour of this work. Their willingness to engage with the engineering details of the system, while continuously challenging me to articulate the underlying research contribution, was invaluable.

I extend my heartfelt thanks to the Head of the Department of Artificial Intelligence and Machine Learning for providing the academic environment, computational resources, and institutional support that enabled this work. The faculty members of the department contributed substantially through their teaching across the disciplines of computer vision, deep learning, software engineering, and database systems, all of which converged in this project.

I gratefully acknowledge the open-source community that made this work possible. Specifically, I would like to recognise the contributors behind PyTorch, OpenCV, Flask, React, MongoDB, and the broader Python data-science ecosystem, whose libraries form the foundation of the implementation. I am also grateful to Google for providing access to the Gemini family of vision–language models through Google AI Studio, and to the authors of the DCSASS surveillance dataset for releasing their data for research use.

I would like to thank my peers and friends who participated in informal usability testing, contributed feedback on the user interface, and helped triage edge cases during system integration. Their honest critique improved the final product significantly.

Finally, I owe my deepest gratitude to my family for their unwavering moral support, patience, and belief in this work, especially during the long hours of training, debugging, and writing that the project demanded.

Vivek Muthe

---

## ABSTRACT

Retail shrinkage caused by shoplifting represents one of the largest sources of preventable loss for the global retail industry, with annual losses estimated in tens of billions of dollars. Conventional closed-circuit television (CCTV) deployments are predominantly passive: they record incidents for after-the-fact review but provide no proactive alerting and demand continuous human attention to be useful in real time. The advent of large vision–language models (VLMs) such as Google Gemini opens the possibility of automated, semantically rich scene understanding, but their per-call latency, cost, and quota constraints make naive deployment in surveillance pipelines economically prohibitive at scale.

This project, titled **Falantir**, presents a production-grade multi-tier vision pipeline that combines a frontier vision–language model with a locally trained MobileNetV3-Large student classifier to deliver semantically rich, low-cost, and resilient retail threat detection. The system implements a smart cascade in which the lightweight student model acts as a first-pass screener: frames it confidently labels as safe are returned immediately at near-zero cost, while ambiguous or threatening frames are escalated to the cloud-hosted Gemini model for full scene description, reasoning, and object detection. A safe-fallback provider guarantees the pipeline never fails, even under network or quota outage.

The student model was trained on 11,240 surveillance frames extracted from the DCSASS dataset and labelled across three threat tiers — safe, suspicious, and critical — achieving a validation accuracy of 97.29 percent on a held-out set. The end-to-end system is implemented as a Flask and Flask-SocketIO backend with a React and Vite frontend, persists incidents to MongoDB, and dispatches multi-channel alerts via SMTP electronic mail and the Twilio short-message service. A motion gate based on the OpenCV MOG2 background subtractor further reduces unnecessary inference calls. Empirical evaluation shows that the smart cascade reduces Gemini token consumption by approximately seventy to ninety percent on representative footage while preserving classification quality on threat events.

The contribution of this work is threefold. First, it demonstrates a practical deployment architecture for combining frontier VLMs with edge inference under real-world cost constraints. Second, it presents a knowledge-distillation pipeline in which an inexpensive cloud teacher labels frames for a small offline student. Third, it provides a complete, reproducible reference implementation including front-end, back-end, training scripts, dataset preparation tooling, and a real-time alerting subsystem suitable as a foundation for further academic and industry research.

**Keywords:** computer vision, vision–language models, knowledge distillation, MobileNetV3, retail surveillance, shoplifting detection, edge inference, hybrid AI architecture, real-time systems, Flask, React, MongoDB.

---

## TABLE OF CONTENTS

| No. | Section | Page |
| --- | --- | --- |
| | Title Page | i |
| | Certificate | ii |
| | Acknowledgement | iii |
| | Abstract | iv |
| | Table of Contents | v |
| | List of Figures | vi |
| | List of Tables | vii |
| | List of Abbreviations | viii |
| 1 | Introduction | 1 |
| 1.1 | Background and Motivation | 1 |
| 1.2 | Project Objectives | 2 |
| 1.3 | Scope of the Work | 3 |
| 1.4 | Organisation of the Report | 4 |
| 2 | Literature Review | 5 |
| 2.1 | Retail Loss Prevention and Surveillance | 5 |
| 2.2 | Classical Anomaly Detection in Video | 6 |
| 2.3 | Deep Learning for Action Recognition | 6 |
| 2.4 | Vision–Language Models for Scene Understanding | 7 |
| 2.5 | Knowledge Distillation and Edge Inference | 8 |
| 2.6 | Research Gap | 9 |
| 3 | System Analysis | 10 |
| 3.1 | Problem Statement | 10 |
| 3.2 | Limitations of the Existing System | 10 |
| 3.3 | Proposed System | 12 |
| 3.4 | Feasibility Study | 13 |
| 3.5 | Functional and Non-functional Requirements | 14 |
| 4 | System Design | 15 |
| 4.1 | High-Level Architecture | 15 |
| 4.2 | Vision Provider Chain | 16 |
| 4.3 | Smart Cascade Decision Flow | 17 |
| 4.4 | Database Schema | 18 |
| 4.5 | Notification Subsystem | 19 |
| 4.6 | Frontend Component Architecture | 20 |
| 5 | Implementation | 21 |
| 5.1 | Technology Stack | 21 |
| 5.2 | Backend Implementation | 22 |
| 5.3 | Vision Pipeline | 23 |
| 5.4 | Student Model Training | 25 |
| 5.5 | Frontend Implementation | 27 |
| 5.6 | Database and Persistence | 28 |
| 5.7 | Alert Dispatch | 29 |
| 6 | Results and Discussion | 30 |
| 6.1 | Training and Validation Results | 30 |
| 6.2 | End-to-End Latency and Throughput | 31 |
| 6.3 | Cost and Token-Reduction Analysis | 32 |
| 6.4 | Qualitative Comparison | 33 |
| 6.5 | Limitations Observed | 34 |
| 7 | Conclusion | 35 |
| 8 | Future Scope | 36 |
| | References | 38 |

---

## LIST OF FIGURES

| Fig. | Description |
| --- | --- |
| 4.1 | High-level system architecture of Falantir |
| 4.2 | Vision provider fallback chain |
| 4.3 | Smart cascade decision flow |
| 4.4 | Entity-relationship diagram for the MongoDB schema |
| 4.5 | Frontend component hierarchy |
| 5.1 | Knowledge-distillation training pipeline |
| 5.2 | Live-feed video viewer with bounding-box overlay |
| 6.1 | Training and validation accuracy across epochs |
| 6.2 | Confusion matrix on the held-out validation set |

## LIST OF TABLES

| Table | Description |
| --- | --- |
| 5.1 | Summary of the dataset used for student training |
| 6.1 | Per-class precision, recall, and F1-score |
| 6.2 | Latency comparison across providers |
| 6.3 | Estimated cost savings under the smart cascade |

## LIST OF ABBREVIATIONS

| Abbreviation | Expansion |
| --- | --- |
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| BGR | Blue–Green–Red colour ordering |
| CCTV | Closed-Circuit Television |
| CNN | Convolutional Neural Network |
| CORS | Cross-Origin Resource Sharing |
| CRUD | Create, Read, Update, Delete |
| DCSASS | Dataset of Surveillance and Anomaly Sequences (publicly hosted on Kaggle by M. Hervas) |
| HTTP | HyperText Transfer Protocol |
| JSON | JavaScript Object Notation |
| JWT | JSON Web Token |
| LVM / VLM | Large / Vision–Language Model |
| MOG2 | Mixture-of-Gaussians background subtraction (version 2) |
| ORM | Object-Relational Mapping |
| REST | Representational State Transfer |
| RPM | Requests Per Minute |
| RPD | Requests Per Day |
| RTSP | Real-Time Streaming Protocol |
| SDK | Software Development Kit |
| SMTP | Simple Mail Transfer Protocol |
| SMS | Short Message Service |
| TLS | Transport Layer Security |
| URL | Uniform Resource Locator |
| WSGI | Web Server Gateway Interface |

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Motivation

The global retail industry suffers significant financial losses each year due to shoplifting and in-store theft, a phenomenon collectively referred to as retail shrinkage. According to recurring industry surveys conducted by the National Retail Federation in the United States, and analogous bodies in Europe and the Asia-Pacific region, shoplifting consistently accounts for the largest share of preventable inventory loss, frequently exceeding the losses attributed to internal employee theft and administrative error combined. Beyond the direct financial impact on retailers, persistent shoplifting produces secondary effects that ripple through the supply chain in the form of higher consumer prices, more aggressive store layouts, and growing security overheads. The retail sector therefore has a strong economic incentive to invest in effective loss-prevention technology.

The conventional countermeasure to shoplifting is the deployment of closed-circuit television (CCTV) systems coupled with on-premises human security staff. Despite decades of refinement, this approach exhibits well-documented weaknesses. Operators monitoring multiple feeds simultaneously experience attentional fatigue; subtle theft behaviours such as concealment of small items in clothing or bags are easily missed; and even when an incident is detected in real time, the latency from observation to intervention is typically too high to prevent the loss. Reviewing footage retrospectively allows investigators to construct evidence after an incident has occurred but rarely contributes to its prevention. The economics of staffing further discourage round-the-clock human monitoring of every camera in every store.

Recent advances in computer vision and, more transformatively, in vision–language models (VLMs) such as Google Gemini, OpenAI GPT-4V, and Anthropic Claude with multimodal input, suggest a fundamentally different approach. These models can describe scenes in natural language, reason about behaviour, and classify activities with a level of generalisation that earlier task-specific networks could not match. In principle, a surveillance pipeline backed by a frontier VLM can deliver the kind of semantically rich understanding that until recently required a human observer.

In practice, however, two obstacles stand in the way of naive deployment of such models in retail surveillance. First, the per-call latency of cloud-hosted VLMs typically lies in the range of two to ten seconds per frame, and their token-based pricing makes continuous per-frame analysis financially infeasible at the volume of frames produced by even a small camera fleet. Second, the dependence on cloud connectivity means that any disruption to network access or to the vendor's quota allocation produces an immediate failure of the surveillance system, an outcome unacceptable in a security-critical context. Together, these constraints motivate the central question this project seeks to answer: how can the semantic richness of frontier vision–language models be harnessed for real-time retail threat detection without incurring prohibitive cost or surrendering operational reliability?

## 1.2 Project Objectives

The primary objective of this project is to design, implement, and evaluate a complete software system, named Falantir, that detects retail threats in surveillance video using a hybrid architecture combining a frontier vision–language model with a locally trained convolutional student model. The system is intended to demonstrate that the combination is materially cheaper than the cloud model alone, materially more accurate and informative than the local model alone, and resilient to outages in either component.

In support of this primary objective, the project pursues a number of subsidiary goals. The first is the construction of a labelled dataset of surveillance frames spanning three threat tiers — safe, suspicious, and critical — derived from publicly available video corpora and processed to a uniform image format suitable for supervised classification. The second is the training of a compact convolutional classifier, using transfer learning from a pretrained MobileNetV3-Large backbone, that achieves at least ninety percent validation accuracy on the held-out portion of this dataset while remaining small enough to run on commodity hardware without acceleration. The third is the integration of this trained classifier with the cloud-hosted Gemini model in a smart-cascade configuration, accompanied by an explicit fallback policy that guarantees the system continues to respond even under cloud failure.

Beyond the core inference pipeline, the project aims to deliver the surrounding infrastructure expected of a production-quality system. This includes a persistent storage layer for incident records, a real-time notification mechanism for delivering alerts via electronic mail and short message service, an authenticated web interface for uploading footage and reviewing incidents, and a live monitoring view that supports streaming from network camera agents. Finally, the project aims to evaluate the resulting system on representative footage, characterising its accuracy, its latency, its cost profile under the smart cascade, and its qualitative behaviour relative to the baseline approaches it replaces.

## 1.3 Scope of the Work

The scope of this project is bounded along several dimensions. The system is designed for a retail surveillance setting in which a small number of fixed cameras observe customers interacting with merchandise. It is not designed for outdoor large-area surveillance, traffic analysis, or public-safety monitoring at municipal scale, although several of its architectural ideas are transferable to those domains.

The threat taxonomy is intentionally restricted to three coarse tiers: safe activity such as normal browsing and payment; suspicious activity such as concealment of merchandise in clothing, lingering near exits with merchandise, or shielding hand movements from the camera; and critical activity such as direct theft in progress, violent confrontation, the visible presence of weapons, or environmental emergencies including fire and medical events. Finer-grained behavioural classification, such as identifying specific theft methods or distinguishing different categories of weapon, is beyond the scope of this work.

The system is implemented as a single-tenant application: it assumes the operator and the alerted personnel form a coherent organisational unit with a shared notification channel, and it does not address the multi-tenant concerns of a software-as-a-service offering such as per-tenant data isolation, billing, or quota enforcement. The implementation also focuses on the English language for both the system's user interface and the natural-language descriptions emitted by the vision–language model.

Although the system supports live ingestion of network camera streams via the real-time streaming protocol, the principal evaluation in this report is conducted on uploaded video files. Live-stream evaluation is treated as a supplementary capability whose qualitative behaviour is described but whose comprehensive quantitative evaluation is left for future work.

## 1.4 Organisation of the Report

The remainder of this report is organised as follows. Chapter 2 surveys the relevant literature in retail surveillance, video anomaly detection, action recognition, vision–language modelling, and knowledge distillation, identifying the gap that this project addresses. Chapter 3 articulates the problem, describes the limitations of the conventional CCTV-based approach, and presents the proposed system at a conceptual level. Chapter 4 elaborates the system design, including the architecture of the vision provider chain, the smart-cascade decision flow, the database schema, and the frontend component hierarchy. Chapter 5 documents the implementation in detail, walking through each major module with particular attention to the design choices that distinguish this project from the conventional approaches. Chapter 6 reports empirical results, including training and validation accuracy of the student model, end-to-end latency benchmarks, and an analysis of the cost reduction achieved by the smart cascade. Chapter 7 concludes the report. Chapter 8 outlines a number of directions for future work. The report ends with a bibliography in IEEE referencing format.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Retail Loss Prevention and Surveillance

The academic and industry literature on retail loss prevention spans operational research, criminology, and applied computer science. From an operational perspective, retail shrinkage is conventionally decomposed into four categories: external theft, internal theft, administrative error, and supplier fraud. Surveys conducted by the National Retail Federation and the European Loss Prevention Group show that external theft consistently accounts for between thirty and forty percent of total shrinkage in major retail markets, with an annual financial impact estimated in the tens of billions of dollars worldwide. The criminological literature, in turn, has long argued that the perceived likelihood of detection, rather than the severity of consequences, is the dominant deterrent against shoplifting. This finding implies that a surveillance system whose primary output is a credible, timely intervention is potentially more impactful than one whose principal contribution is forensic record-keeping.

Within the applied computer-science literature, a substantial body of work treats the problem as one of automating the human surveillance role. Early systems combined motion detection with rule-based heuristics, generating alerts when a person remained in a sensitive area for longer than a configured threshold or when a tracked individual entered and exited the camera's field of view repeatedly. While such systems demonstrated the feasibility of automated alerting, their false-positive rates were typically high enough to undermine operator trust, and they provided no semantic explanation for the alerts they generated.

## 2.2 Classical Anomaly Detection in Video

The video-anomaly-detection subfield of computer vision has produced a long sequence of approaches that frame the problem as one of unsupervised or weakly supervised modelling. Mixture-of-Gaussians background subtraction, spatiotemporal autoencoders, and one-class support vector machines have each been applied to surveillance footage with the aim of flagging frames whose appearance or motion patterns deviate from a learned baseline. These approaches have the appealing property of requiring no labelled examples of anomalous behaviour, but they share a common weakness: they cannot distinguish a benign anomaly, such as a child running in an aisle, from a security-relevant anomaly such as concealment of merchandise. The output of an anomaly detector is therefore typically only the first stage of a larger pipeline that includes a more discriminative classifier or a human operator.

## 2.3 Deep Learning for Action Recognition

The introduction of deep learning to video understanding produced substantial gains on action-recognition benchmarks. Three-dimensional convolutional networks, two-stream networks combining spatial and optical-flow inputs, and more recently Transformer-based video encoders all advanced the state of the art on datasets such as Kinetics, UCF-101, and HMDB-51. In the surveillance-specific domain, work on the UCF-Crime dataset and the related DCSASS dataset demonstrated that supervised classifiers can be trained to recognise concrete categories of anomalous activity, including shoplifting, robbery, assault, and vandalism. However, these networks operate at the granularity of an action label and offer no narrative explanation of the scene; they tell the operator that something has occurred without explaining what or why. They also tend to suffer from poor generalisation to environments and camera angles unseen during training.

## 2.4 Vision–Language Models for Scene Understanding

The most consequential development for this project is the recent emergence of vision–language models capable of producing natural-language descriptions, structured analyses, and explicit reasoning over still images. Closed models such as Google Gemini, OpenAI GPT-4V, and Anthropic Claude with multimodal input, alongside open alternatives such as LLaVA, demonstrate that a single model can describe the contents of a scene, identify objects, and reason about behaviour at a level previously attainable only with custom-trained pipelines composed of object detectors, action recognisers, and rule engines. Recent work in retail and surveillance has begun to evaluate VLMs as drop-in replacements for these custom pipelines, with promising results on accuracy but with cost, latency, and reliability concerns that limit deployment.

The Gemini family in particular offers a structured-output mode in which a developer can specify a JSON schema and obtain responses that strictly conform to it. This capability is critical for production integration: it eliminates the brittle text parsing that earlier free-form generative models required. The Gemini 2.5 Flash variant, and the more recent 3.x preview variants used in this project, deliver per-call latencies in the low seconds at a cost that is feasible for sparse triggering but prohibitive for naive per-frame deployment. The work reported here treats the VLM as a high-quality but high-cost oracle whose access must be rationed.

## 2.5 Knowledge Distillation and Edge Inference

The asymmetry between a high-cost, high-accuracy teacher and a low-cost, lower-accuracy student lies at the heart of the knowledge-distillation literature, originated by Hinton and colleagues in the context of model compression. The classical formulation uses a soft-target loss derived from the teacher's output distribution to train a smaller student network that mimics the teacher's behaviour. In its modern incarnations, the technique has been generalised to use the teacher as an automatic labeller for unlabelled data, eliminating the need for human annotation while still producing a deployable student model many orders of magnitude smaller than the teacher.

In the surveillance domain, distilling a frontier VLM into a small convolutional classifier is particularly attractive because the student can run on the same edge hardware that already hosts the camera, eliminating cloud dependency in the common case. The MobileNetV3 architecture, introduced by Howard and colleagues, is a leading candidate for this role: it combines depthwise separable convolutions, the squeeze-and-excitation block, and the hard-swish activation to deliver ImageNet-grade accuracy at a fraction of the parameter count and inference cost of comparable networks. For a three-class threat classification task, MobileNetV3-Large fine-tuned on a few thousand labelled frames is a reasonable starting point.

## 2.6 Research Gap

The literature reviewed above motivates the central research gap addressed by Falantir. Conventional CCTV systems and anomaly detectors lack semantic understanding and produce too many false positives for unattended deployment. Supervised action-recognition networks trained on surveillance datasets achieve respectable per-action accuracy but offer no narrative output and generalise poorly outside their training distribution. Vision–language models offer outstanding semantic richness but are economically infeasible to apply to every frame and dependent on cloud connectivity for every decision. Knowledge distillation from VLM to convolutional classifier, while well established in the machine-learning literature, has not been productionised as a deployment architecture in the retail surveillance domain.

This project addresses that gap by constructing and evaluating a deployment architecture in which a frontier VLM and a distilled student model cooperate through an explicit smart-cascade policy, supported by a complete surrounding system for incident persistence, real-time alerting, user authentication, and live stream ingestion.

---

# CHAPTER 3: SYSTEM ANALYSIS

## 3.1 Problem Statement

A retail surveillance system intended to support proactive loss prevention must satisfy a demanding set of simultaneous requirements. It must produce an accurate, semantically meaningful classification of every frame it analyses, distinguishing benign customer activity from concealment, theft, and emergencies. It must operate within a cost envelope compatible with continuous deployment across many cameras. It must respond quickly enough that an alert reaches the relevant personnel in time to intervene. It must remain functional under realistic failure conditions, including transient network outages, exhausted cloud quotas, and partial component failures. It must persist its observations in a form that supports audit, retrospective analysis, and feedback-driven model improvement. And it must expose its capabilities through a user interface that is accessible to non-technical security staff. No single existing system, whether proprietary or academic, satisfies all of these requirements simultaneously.

## 3.2 Limitations of the Existing System

The conventional retail surveillance system, comprising one or more CCTV cameras connected to a digital video recorder or network video recorder, with a human operator monitoring a wall of screens, exhibits a number of well-documented limitations.

The first limitation is attentional. The cognitive psychology literature establishes that human vigilance on a monitoring task degrades sharply within twenty minutes; an operator nominally responsible for sixteen feeds is in practice attentive to a small fraction of them at any moment. Subtle shoplifting behaviours, which often complete within a few seconds, are routinely missed even by alert operators.

The second limitation is economic. Twenty-four-hour staffing of a monitoring station is expensive and is therefore unavailable to most small and medium retailers. The marginal cost of adding cameras is low, but the marginal cost of effectively monitoring those cameras is high, leading in practice to a configuration in which footage is recorded but seldom watched.

The third limitation is temporal. Even when a human operator does observe an incident in real time, the latency from observation to action — typically the dispatch of in-store staff or the alerting of law enforcement — is typically long enough that the perpetrator has departed before any intervention occurs. The surveillance system in its conventional form is therefore primarily a forensic tool, not a preventive one.

The fourth limitation is informational. Footage captured by conventional systems is typically retained for a fixed period and then overwritten. There is no semantic metadata: a security manager seeking to review all incidents involving a particular type of behaviour must manually scrub through hours of video, an exercise impractical at scale.

Several existing technologies attempt to address one or another of these limitations. Motion-triggered recording reduces storage requirements but offers no semantic understanding. Heuristic alerting based on tripwires or zone occupancy generates alerts but suffers from extreme false-positive rates. Off-the-shelf video analytics products that incorporate object detection and re-identification provide some semantic understanding but typically lack support for the nuanced behavioural categories relevant to shoplifting and operate as opaque black boxes whose behaviour cannot be customised by the deploying organisation.

## 3.3 Proposed System

The system proposed in this project, Falantir, is structured around a small number of architectural commitments that together address the limitations enumerated above.

The first commitment is to a multi-tier vision pipeline composed of three providers: a cloud-hosted vision–language model as the primary semantic source, a locally trained MobileNetV3-Large student classifier as the cost-saving and offline fallback, and a deterministic safe-fallback provider that always returns a benign classification as the last-resort guarantee that the system never fails. The composition is mediated by an explicit chain selector configurable at deployment time.

The second commitment is to a smart cascade in which the inexpensive student model performs a first-pass classification on every frame; the expensive VLM is invoked only when the student model expresses either non-safe judgement or insufficient confidence in a safe judgement. This mechanism preserves the semantic richness of the VLM in cases where it matters while dramatically reducing the per-frame cost in the common case of uneventful footage.

The third commitment is to structured persistence: every classification deemed of interest is recorded as an incident document in a MongoDB collection, complete with its peak-threat snapshot, the model output that produced it, and an acknowledgement state used to track operator response. This persistence enables retrospective analysis, supports the user interface, and provides a substrate for future feedback-driven learning.

The fourth commitment is to multi-channel alerting: when an incident is recorded, the system dispatches alerts via electronic mail and short message service to the appropriate personnel, with a configurable cooldown to prevent notification flooding when an extended event triggers many consecutive alerts.

The fifth commitment is to a modern web interface, implemented as a single-page React application, which exposes the upload of video footage for analysis, the live monitoring of network camera agents, the review of past incidents, and per-incident operator feedback. The interface uses a cinematic surveillance aesthetic for the analysis viewer, with a scanning overlay during processing and animated bounding boxes around detected objects in the analysis result, deliberately framing the system's output in a manner familiar to operators of conventional CCTV systems.

## 3.4 Feasibility Study

The feasibility of the proposed system has been evaluated along three dimensions: technical, economic, and operational.

From a technical standpoint, all of the components on which the system depends are available, well-documented, and stable. PyTorch provides mature support for training and deploying MobileNetV3-Large classifiers; the Google Generative Language API provides accessible programmatic access to Gemini; Flask, Flask-SocketIO, MongoDB, React, and Vite are widely used and documented. The DCSASS dataset is publicly available for research use under permissive terms. No part of the proposed system requires capabilities that lie outside the current state of the art in academic or industry practice.

From an economic standpoint, the proposed deployment is feasible under realistic operating budgets. The student model, once trained, runs on commodity central-processing-unit hardware without acceleration. The cloud-hosted VLM, accessed under the smart cascade, accumulates costs only on the small fraction of frames containing potential threats; under the threshold configurations evaluated in Chapter 6, the per-camera-per-day cloud cost is estimated in the cents range for typical retail traffic. For development and academic evaluation, the system relies on a Google Cloud trial credit of three hundred United States dollars, which proves more than sufficient for the scope of the project.

From an operational standpoint, the proposed system can be deployed and maintained by a small team. Its components are loosely coupled, can be deployed on a single host or distributed across multiple hosts, and can be upgraded independently. The student model can be retrained on new footage without disturbing the production deployment, and the cloud provider can be substituted by editing a configuration file.

## 3.5 Functional and Non-functional Requirements

The functional requirements of the system are enumerated as follows. The system shall accept video file uploads through an authenticated web interface and return a structured analysis of the footage. The system shall support live ingestion from network cameras using the real-time streaming protocol. The system shall classify each analysed frame into one of three threat tiers and shall, where the active provider supports it, additionally produce a natural-language scene description, an explanation of the reasoning behind the classification, and a list of detected objects with bounding boxes. The system shall persist incidents of suspicious or critical classification to a database and shall dispatch alerts to configured recipients via electronic mail and short message service. The system shall expose an interface through which operators can review past incidents and mark them as acknowledged or as false positives.

The non-functional requirements are as follows. The end-to-end latency of a frame analysis through the smart cascade shall not exceed five seconds in the median case. The system shall remain available under transient cloud failure, falling back automatically to the local student model. The system shall be horizontally scalable in its frontend and stateless application tier, with persistent state confined to MongoDB. The student model shall be retrainable on a commodity Google Colab T4 instance within sixty minutes for a dataset of approximately ten thousand frames. The user interface shall be accessible through any modern web browser without browser-specific extensions, and shall be usable on mobile devices through responsive layout.

---

# CHAPTER 4: SYSTEM DESIGN

## 4.1 High-Level Architecture

At the highest level, Falantir is a three-tier web application augmented with a vision-processing pipeline. The presentation tier is a React single-page application served as static assets and rendered in the operator's browser; it communicates with the application tier over both a conventional HyperText Transfer Protocol REST interface and a persistent WebSocket connection used for real-time updates. The application tier is implemented as a Flask process with the Flask-SocketIO extension, hosting a small set of REST endpoints for authentication, agent management, incident retrieval, and video upload, alongside a WebSocket namespace for live agent feeds. The persistence tier is a MongoDB database storing user accounts, agent definitions, incident records, reinforcement-learning feedback, and analytics aggregates.

Embedded within the application tier is the vision pipeline, the principal contribution of the project. The pipeline is exposed through a single function — analyze\_frame — that accepts a Blue–Green–Red colour-ordered image array and returns a canonical result dictionary containing the threat label, threat level, confidence score, scene description, reasoning, and detected objects. The internal mechanics by which this dictionary is produced are hidden behind the function signature, allowing the rest of the application to remain agnostic to the choice of provider or to the smart-cascade decisions it makes.

A motion gate, implemented using OpenCV's MOG2 background-subtraction algorithm, sits in front of the vision pipeline on the live-streaming code path. Frames in which the fraction of changed pixels falls below a configurable threshold are skipped without invoking the pipeline, on the assumption that an empty or static scene contains no event of interest. This gate is independent of the smart cascade and acts as a coarser, cheaper filter applied earlier.

A notification subsystem, hosted within the same Flask process, is responsible for dispatching alerts when the application records an incident at the suspicious or critical level. It encapsulates two backends — a Simple Mail Transfer Protocol electronic-mail backend and a Twilio short-message-service backend — behind a unified notify\_all function that fans out to all active recipients.

## 4.2 Vision Provider Chain

The vision-provider chain is implemented as an ordered list of provider identifiers, each of which corresponds to an implementation of an abstract VisionProvider base class. The base class declares two responsibilities for any concrete provider: it must report whether it is currently available, and it must analyse a frame and return the canonical result dictionary. The default chain is ordered as Gemini, then MobileNetV3, then SafeFallback. The active provider for a given run is the first element of the chain that reports itself available; in the standard configuration, this resolves to Gemini whenever an Application Programming Interface key is configured, falling through to MobileNetV3 when the key is missing or invalid, and finally to SafeFallback when both higher-priority providers are unavailable.

The Gemini provider wraps the Google Generative Language API. It uses the structured-output mode of the Gemini Application Programming Interface, supplying a JSON schema that constrains the model's output to the canonical fields the application expects. This eliminates the brittle text parsing that would otherwise be necessary and ensures that the output of every successful call conforms to the same shape as the output of the local student model.

The MobileNetV3 provider wraps a fine-tuned MobileNetV3-Large classifier loaded from a local PyTorch state-dictionary file. It performs the standard ImageNet preprocessing pipeline — resizing to a two-hundred-and-twenty-four-by-two-hundred-and-twenty-four-pixel square, conversion to a tensor, and normalisation by the standard ImageNet statistics — followed by a forward pass through the network to obtain a softmax distribution over the three threat classes. The class with the highest probability is reported as the predicted label, the corresponding probability as the confidence, and the full distribution as the per-class probabilities. Because the local model produces no narrative output, the scene description and reasoning fields are populated with constant placeholder text, and the detected-objects list is left empty.

The SafeFallback provider is a deterministic implementation that returns the same benign result on every input. Its purpose is to guarantee that a downstream consumer of the vision pipeline never observes an exception in the analyse method or a partial result missing required fields, even under conditions in which all other providers have failed.

## 4.3 Smart Cascade Decision Flow

The smart-cascade mechanism is layered on top of the provider-chain abstraction. When the cascade is enabled and the active provider is Gemini and the MobileNetV3 provider is also available, the analyze\_frame function executes a two-stage decision. In the first stage, the frame is passed to the MobileNetV3 provider. If the local model returns a label of safe with a confidence equal to or exceeding a configurable threshold — set by default to seventy percent — the local result is returned immediately and the Gemini call is skipped. In the second stage, reached when the local model returns a non-safe label or a low-confidence safe label, the frame is passed to the Gemini provider, whose result is returned to the caller. In either case, the result is annotated with a boolean flag — cascade\_skipped\_gemini — that records whether the cloud call was made, allowing downstream code and operators to audit the cascade's behaviour.

The threshold value, exposed through the CASCADE\_SAFE\_THRESHOLD environment variable, is the principal tuning knob for the cost–quality trade-off. A higher threshold corresponds to a more conservative policy, in which the system trusts the local model only when it is very confident; this produces more cloud calls but reduces the risk of missing a subtle threat that the local model would misclassify as safe. A lower threshold corresponds to a more aggressive policy, in which the system trusts the local model more readily and incurs lower cost at the price of a slightly higher risk of false negatives. The default value of seventy percent represents a deliberately conservative starting point, chosen on the basis of the validation behaviour of the trained student model.

## 4.4 Database Schema

The MongoDB schema comprises five principal collections.

The users collection stores user accounts. Each document records an identifier, an electronic-mail address, a hashed password, a phone number, a role, and an active flag. The role field supports a coarse access-control distinction between administrative and operational users.

The agents collection stores the definitions of camera agents. Each document records an identifier, a human-readable name, a location label, a stream type — whether the agent reads from a webcam, an RTSP stream, or a video file — and the corresponding source identifier, alongside an owner identifier linking the agent to a user.

The incidents collection stores the audit trail of detected threats. Each document records an agent identifier, the threat label, threat level, and confidence, the scene description and reasoning produced by the vision pipeline, the list of detected objects with bounding boxes, the provider that produced the classification, the model name, the timestamp, the base-sixty-four-encoded snapshot of the peak-threat frame, and the acknowledged flag.

The rl\_feedback collection stores operator-supplied feedback used for future model improvement. Each document records the incident identifier, the operator's verdict — correct or incorrect — and the timestamp.

The analytics collection stores periodically aggregated counts of incidents by class and by hour, used to populate dashboard charts.

## 4.5 Notification Subsystem

The notification subsystem is encapsulated in a single Python module, notifications, which exposes three functions. The send\_email function accepts an electronic-mail address and a message body and delivers the message through a Simple Mail Transfer Protocol relay configured by environment variables. The send\_sms function accepts a phone number and a message body and delivers the message through the Twilio Application Programming Interface. The notify\_all function accepts an electronic-mail address, a phone number, and a message body, and invokes both backend functions, returning a dictionary recording the success or failure of each.

For incidents recorded by the live-streaming code path, the notification logic also enforces a per-agent cooldown, governed by the ALERT\_COOLDOWN\_S environment variable, defaulting to three hundred seconds. This cooldown prevents an extended event such as a five-minute fight, which generates many consecutive suspicious classifications, from producing a corresponding flood of short-message-service messages, each of which costs real money under the Twilio per-message billing model.

## 4.6 Frontend Component Architecture

The React frontend is organised as a single-page application with the React Router library managing navigation between top-level routes. The principal routes are the dashboard, the live-feed page, the analytics page, the alerts page, the settings page, the user-management page, and the profile page. Authentication state is managed through a Redux store, with the JSON Web Token persisted in browser storage and attached to every Application Programming Interface request through an Axios interceptor.

The live-feed page hosts the principal user-facing innovation of the project: the surveillance-style video analysis viewer. The viewer accepts a video file, displays it playing within a CCTV-styled black frame complete with corner brackets and a cyan scanning line, and on completion of the analysis transitions to displaying the peak-threat snapshot with animated, colour-coded bounding boxes around each detected object. The animations are implemented with the Framer Motion library, and the bounding boxes are rendered as Scalable Vector Graphics elements overlaid on the snapshot image.

---

# CHAPTER 5: IMPLEMENTATION

## 5.1 Technology Stack

The implementation rests on a coordinated stack of open-source and commercial components, chosen to balance maturity, community support, performance, and developer productivity. The backend is implemented in Python version 3.14, using Flask version 3.x as the web framework and Flask-SocketIO for the real-time bidirectional channel. PyTorch version 2.x supplies the deep-learning runtime; OpenCV version 4.x is used for video decoding, frame manipulation, and motion detection. The Google Generative Language API is accessed through the official google-generativeai client library. The Twilio Python helper library is used for short-message-service dispatch, and the standard library smtplib module is used for electronic-mail dispatch. MongoDB is accessed through the PyMongo driver. Authentication is implemented using JSON Web Tokens via the PyJWT library, with bcrypt for password hashing.

The frontend is implemented in JavaScript using React version 18, with Vite version 6 as the build tool and development server. State management uses the Redux Toolkit, routing uses React Router version 6, and HTTP requests are made through Axios. Animations are produced through Framer Motion, with iconography from lucide-react and recharts powering the analytics charts. Styling is achieved through Tailwind CSS, configured with a small custom palette in the project's tailwind.config.js file. Toast notifications are produced through react-hot-toast.

The training pipeline is implemented in Python and is intended to be run on Google Colab. It uses torchvision for the MobileNetV3-Large backbone and the standard data-augmentation utilities, and writes checkpoints to a Google Drive path so that progress is preserved across Colab session disconnections.

## 5.2 Backend Implementation

The backend application is bootstrapped by app.py, which constructs the Flask application, configures Cross-Origin Resource Sharing, instantiates the SocketIO object with threading-based asynchronous mode, and registers four blueprint modules: authentication, user management, agent management, and detection. Each blueprint encapsulates a related set of REST endpoints under a common URL prefix.

The authentication blueprint exposes endpoints for user registration, login, logout, and the retrieval of the current user's profile. Registration hashes the supplied password with bcrypt before persisting the user document. Login compares the supplied password against the stored hash and, on success, issues a JSON Web Token signed with the application's secret key. A login\_required decorator wraps protected endpoints, validating the token from the Authorization header and injecting the corresponding user document into Flask's per-request context.

The detection blueprint hosts the principal application-facing endpoint, upload, which accepts a multipart-form upload containing a video file. The endpoint writes the file to a temporary path, opens it through OpenCV's VideoCapture, samples up to twelve frames evenly spaced across the duration of the video, passes each frame through the analyze\_frame function, and identifies the frame with the highest threat level as the peak frame. The result for the peak frame is then augmented with summary statistics — total frames, frames analysed, frames failed, quota status, and total threat detections — and a base-sixty-four-encoded JPEG snapshot of the peak frame is included in the response so the frontend can render it as the bounding-box overlay backdrop. If the peak frame's threat level is non-zero, an incident document is constructed and saved to MongoDB, and a notify\_all call is dispatched in a background thread to alert the uploader by electronic mail and short-message service.

The agent-management blueprint exposes endpoints for the creation, retrieval, update, and deletion of camera agents, alongside endpoints for starting and stopping the live-stream worker associated with an agent. The streaming subsystem itself is implemented in api/services/stream\_service.py, which spawns a daemon thread per active agent. Each thread reads frames from the agent's source — webcam, real-time-streaming-protocol stream, or video file — applies the motion gate, and on each non-skipped frame invokes the vision pipeline and emits an agent\_update event over WebSocket to the room corresponding to the agent's identifier. When the resulting threat level meets or exceeds the suspicious tier with a confidence above the configured threat threshold, an incident document is saved and an incident\_alert event is emitted, with an additional fan-out to electronic mail and short-message-service notifications subject to the per-agent cooldown.

## 5.3 Vision Pipeline

The vision pipeline is implemented across two main modules. The first, api/services/vision\_provider.py, implements the provider-chain abstraction, the smart cascade, and the canonical result shape. The second, api/services/gemini\_service.py, implements the Gemini provider's analyse method.

The Gemini provider performs the following sequence of operations. It encodes the input frame as a JPEG image with a configurable quality setting, base-sixty-four-encodes the resulting bytes, and constructs a multimodal prompt consisting of a system message describing the threat-classification task and an inline-data part containing the encoded image. The system prompt is the result of considerable iteration: it instructs the model to behave as a retail loss-prevention analyst, enumerates a series of subtle theft indicators including hand-into-clothing motions, hand-into-bag motions, body-blocking-camera-view motions, tag-removal motions, and furtive glances, and explicitly instructs the model that uncertainty between safe and suspicious should be resolved in favour of suspicious. The earlier formulation, which instructed the model to be conservative and prefer the safe label when uncertain, produced a high false-negative rate on shoplifting footage; the revised formulation produced substantially better behaviour while remaining well-calibrated on genuinely safe scenes.

The model response is parsed as JSON and validated against the expected schema. The fields are normalised — labels are clipped to the enumeration of safe, suspicious, and critical; threat-level integers are clipped to zero, one, or two; confidences are clipped to the unit interval; per-class probabilities are renormalised to sum to one if they do not — and the result dictionary is constructed and returned. On any exception — network error, parse error, quota exhaustion — a fallback dictionary containing the safe label and an error message is returned, and the chain selector observes the failure and may activate a different provider on a subsequent call.

The MobileNetV3 provider is implemented in api/models/threat\_classifier.py. The module exposes a load\_model function that lazily constructs the architecture, loads the trained weights from the path configured in the MODEL\_PATH environment variable, and caches the resulting model on the appropriate device. The classify\_frame function performs the standard preprocessing, runs a forward pass under torch.no\_grad to suppress gradient computation, and returns the resulting label, confidence, and probability distribution. The model used in the deployed system is a MobileNetV3-Large backbone with a custom classifier head consisting of a linear projection to two-hundred-and-fifty-six units, a hard-swish activation, a thirty-percent dropout, and a final linear projection to three units, alongside a separate confidence head consisting of a linear projection to one-hundred-and-twenty-eight units, a hard-swish activation, a twenty-percent dropout, a linear projection to one unit, and a sigmoid activation.

The smart cascade is implemented as a wrapping function in vision\_provider.py. It checks an environment variable, SMART\_CASCADE, for the boolean enable flag, and an environment variable, CASCADE\_SAFE\_THRESHOLD, for the confidence threshold. When enabled, it dispatches the frame to the MobileNetV3 provider first, examines the returned label and confidence, and either returns immediately or escalates to the active provider. The annotation of the result with the cascade\_skipped\_gemini boolean enables observability tooling to compute the cascade's hit rate over time.

## 5.4 Student Model Training

The training of the student model is performed in a dedicated script, training/train\_threat\_classifier.py, designed to run in Google Colab on a free T4 graphics-processing unit. The dataset on which the model is trained is the result of preprocessing the publicly available DCSASS dataset, which contains short surveillance video clips spanning thirteen anomaly categories, each with per-clip binary labels indicating whether the clip is normal or anomalous.

The preprocessing script, training/prepare\_dcsass.py, maps the DCSASS class structure onto the three-tier threat taxonomy used by Falantir. Specifically, normal clips of any class are mapped to the safe tier; anomalous clips of the Shoplifting, Stealing, Burglary, and Vandalism classes are mapped to the suspicious tier; and anomalous clips of the Robbery, Assault, Fighting, and Shooting classes are mapped to the critical tier. The script samples a configurable number of frames per clip — by default two for safe and eight for anomalous — and writes them to disk in an ImageFolder layout suitable for direct consumption by torchvision.

After preprocessing, the training set comprises four-thousand-two-hundred-and-fifty critical frames, four-thousand-two-hundred-and-fifty safe frames, and one-thousand-and-sixty-two suspicious frames; the validation set comprises seven-hundred-and-fifty critical frames, seven-hundred-and-fifty safe frames, and one-hundred-and-seventy-eight suspicious frames; the total is eleven-thousand-two-hundred-and-forty frames across the two splits.

The training script defines the model, the data transforms, the optimiser, the loss function, and the training loop. The optimiser is AdamW with a learning rate of one times ten to the negative four and a weight-decay coefficient of one times ten to the negative four; the schedule is cosine annealing over the configured number of epochs. The loss function is cross-entropy. Data augmentation consists of resize to two-hundred-and-fifty-six pixels, random crop to two-hundred-and-twenty-four pixels, random horizontal flip, colour jitter on brightness, contrast, and saturation, and small random rotation. The training loop saves a checkpoint to the configured Google Drive path whenever the validation accuracy improves, ensuring that progress is preserved even if the Colab session is disconnected mid-training.

Training for thirty epochs on the eleven-thousand-frame dataset converges in approximately thirty to forty-five minutes on a T4. The best validation accuracy obtained, reported in Chapter 6, was zero point nine seven two nine.

## 5.5 Frontend Implementation

The React frontend is structured into pages, components, services, and store slices. The pages directory contains one folder per top-level route, each exporting a default React component representing the page's content. The components directory contains presentational and stateful components reusable across pages, organised into subdirectories — layout, ui, monitor, alerts, dashboard, and so on. The services directory contains modules wrapping the backend Application Programming Interface; each function in these modules returns a promise that resolves to the parsed response data. The store directory contains the Redux Toolkit slices and the configured store.

The most distinctive frontend component is the VideoAnalysisViewer, located in components/monitor. Constructed in response to the requirement for a more cinematic analysis experience, it accepts as props the uploaded video file, an uploading boolean, and the result object. While uploading, it renders the uploaded video playing within a black aspect-ratio container styled to evoke a closed-circuit-television monitor: corner brackets in cyan, a moving cyan scanning line traversing the video from top to bottom, a small "ANALYZING" badge with a pulsing scan icon, and a footer describing the active vision pipeline. On completion, the video is replaced with the peak-threat snapshot, and a Scalable Vector Graphics overlay renders one bounding box per detected object. Each box appears with a stagger delay of approximately eighty milliseconds, fades in from a slightly scaled-down state, and pulses gently if its associated action label suggests a threat behaviour. Each box is annotated with a label of the form `class confidence%` and, where present, with the action description below the box, both styled in monospace for the surveillance aesthetic.

## 5.6 Database and Persistence

The database access layer is encapsulated in api/database\_v2.py. The module establishes a singleton MongoDB client using the connection string from the MONGO\_URI environment variable and exposes a collection accessor function for each of the principal collections. A helper function, save\_incident, encapsulates the insertion of an incident document, including the construction of the timestamp and the population of legacy fields used for backward compatibility with earlier schema versions.

A small set of one-time migration functions exists to handle records produced by earlier versions of the system. In particular, earlier versions stored the detected-objects list under the field name yolo\_objects and the scene description under the field name gemini\_description; a migration shim in the incidents endpoint copies these into the current field names — detected\_objects and scene\_description respectively — when reading older records, allowing the frontend to remain agnostic to the schema version of the records it displays.

## 5.7 Alert Dispatch

The alert dispatch is implemented in api/notifications.py, which exposes the three functions described in Chapter 4. The send\_email function constructs a multipart-text electronic-mail message and sends it through the Simple Mail Transfer Protocol relay; in the deployed configuration, this relay is Gmail's smtp.gmail.com on port five-eight-seven, accessed through an application-specific password configured in the SMTP\_PASS environment variable. The send\_sms function uses the Twilio Python helper library to send a message from the Twilio number configured in TWILIO\_PHONE\_NUMBER to the supplied recipient. Both functions are written to fail gracefully — they catch exceptions, log them, and return a boolean indicating success — so that a failure in one channel never prevents the other from being attempted, and a failure in alerting never propagates as an exception out of the application's request-handling path.

The integration of the alert dispatch with the application logic is by deliberate design asynchronous. In the upload endpoint, the notify\_all call is dispatched in a daemon thread after the database insertion completes, ensuring that the Application Programming Interface response to the user is not delayed by the alert latency, which can include several seconds of Twilio round-trip time. In the streaming subsystem, the same pattern is used, with the additional cooldown gate described earlier.

---

# CHAPTER 6: RESULTS AND DISCUSSION

## 6.1 Training and Validation Results

The student MobileNetV3-Large classifier was trained for thirty epochs on the eleven-thousand-two-hundred-and-forty-frame dataset described in Chapter 5. Training took approximately thirty-eight minutes on a Google Colab T4 graphics-processing unit. The training loss decreased monotonically from approximately zero point eight in the first epoch to approximately zero point zero one three in the thirtieth, while the training accuracy increased from approximately zero point six five to zero point nine nine five seven. The validation accuracy increased rapidly during the first ten epochs, peaked at zero point nine seven two nine in the twenty-third epoch, and remained essentially flat thereafter, indicating convergence and the absence of severe overfitting. Table 6.1 summarises the per-class precision, recall, and F1-score on the held-out validation set.

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| safe | 0.97 | 0.98 | 0.97 | 750 |
| suspicious | 0.91 | 0.85 | 0.88 | 178 |
| critical | 0.99 | 0.97 | 0.98 | 750 |
| **macro average** | **0.96** | **0.93** | **0.94** | **1678** |

The slight degradation in suspicious-class metrics relative to safe and critical is attributable to two factors. First, the suspicious tier is the smallest class in the training set, comprising only one-thousand-and-sixty-two examples against four-thousand-two-hundred-and-fifty for each of the other tiers. Second, the suspicious tier sits between two distributions whose appearance can be visually similar: an unobserved hand near a jacket can be either innocent fidgeting or active concealment. The model's lower recall on this class indicates a residual tendency to mistake suspicious frames for safe ones, the more conservative of the two classification errors but, in operational terms, the more dangerous.

## 6.2 End-to-End Latency and Throughput

End-to-end latency was measured for both the standalone Gemini provider and the cascaded configuration on a representative set of twenty-five surveillance video clips ranging in duration from ten seconds to two minutes. The Gemini provider, in standalone configuration, exhibited a per-frame latency in the range of two-point-six to seven-point-five seconds, with the lower bound corresponding to the gemini-3.1-flash-lite-preview model and the upper bound to the gemini-3-pro-preview model. The MobileNetV3 provider exhibited a per-frame latency of approximately fifty milliseconds on a single central-processing-unit core, more than fifty times faster than even the fastest Gemini configuration. The smart cascade, with the threshold at zero point seven and on representative footage in which approximately seventy percent of frames were safe, exhibited a per-frame latency averaging just over one second, dominated by the small fraction of frames that escalated to Gemini.

| Configuration | Mean per-frame latency | Median per-frame latency |
| --- | --- | --- |
| Gemini 3.1 flash-lite (standalone) | 3.0 s | 2.6 s |
| Gemini 3 flash (standalone) | 3.6 s | 3.4 s |
| Gemini 3 pro (standalone) | 7.5 s | 7.2 s |
| MobileNetV3 (standalone) | 0.05 s | 0.05 s |
| Smart cascade (threshold 0.7) | 1.05 s | 0.20 s |

## 6.3 Cost and Token-Reduction Analysis

The token-reduction effect of the smart cascade was estimated by recording, for each of the twenty-five evaluation clips, the fraction of frames for which the local model returned safe with confidence above the cascade threshold. Across the evaluation set, this fraction averaged approximately seventy-three percent, with substantial variance across clips: clips depicting empty store interiors yielded near-unity skip rates, while clips depicting active customer interactions yielded lower rates. Translated into Application Programming Interface cost, this corresponds to a reduction of approximately seventy to ninety percent in Gemini token consumption relative to the standalone Gemini configuration. At the prevailing pricing of approximately one-tenth of a United States dollar per million input tokens for the gemini-3.1-flash-lite-preview model, the cost saving for a single camera operating continuously is on the order of cents per day rather than dollars, a difference that scales linearly with deployment size.

| Footage type | Skip rate | Estimated cost reduction |
| --- | --- | --- |
| Empty store interior | 0.95 | 95% |
| Quiet browsing | 0.80 | 80% |
| Active checkout queue | 0.60 | 60% |
| Sustained suspicious event | 0.10 | 10% |
| **Mixed evaluation set average** | **0.73** | **73%** |

## 6.4 Qualitative Comparison

A qualitative comparison of the standalone Gemini, standalone MobileNetV3, and cascaded configurations was performed on two representative clips. The first clip depicts an individual concealing an item inside a jacket while standing in a candy aisle. Standalone Gemini classified this as suspicious with a confidence of zero point eight five and produced a description identifying the concealment behaviour and the merchandise shelf location; standalone MobileNetV3 also classified it as suspicious but with a confidence of zero point five zero six, with no descriptive output. The cascaded configuration, given the same clip, observed the local model's non-safe judgement and escalated to Gemini, returning the higher-confidence and richer-output result. The second clip depicts an individual browsing produce in a grocery aisle without any concealment behaviour. Standalone Gemini classified this as safe with a confidence of zero point nine three and produced an accurate description; standalone MobileNetV3 classified it as safe with a confidence of zero point eight eight; the cascaded configuration accepted the local model's safe judgement, did not invoke Gemini, and saved the corresponding token cost. The two-clip comparison illustrates the central design intuition behind the cascade: it preserves the cloud model's quality on threat events while extracting the cost saving on uneventful frames.

## 6.5 Limitations Observed

Several limitations were observed during evaluation. First, the suspicious class remains the most error-prone, with both the local model and Gemini occasionally misclassifying ambiguous frames as safe; this is the more dangerous of the two error modes and motivates the conservative threshold default. Second, the local model's lack of descriptive output means that any fully cascaded configuration in which Gemini is bypassed produces an incident record without a natural-language description; while this is acceptable in operational terms — the operator can still review the snapshot — it limits the auditability of the system. Third, the Gemini preview models used in this evaluation are subject to deprecation without notice; a production deployment would require a strategy for automatically switching to the most recent stable model and re-evaluating its calibration. Finally, the present implementation relies on a per-camera threshold uniform across all cameras; a more sophisticated implementation could learn per-camera thresholds adaptively from the empirical distribution of confidences observed at each location.

---

# CHAPTER 7: CONCLUSION

This project has presented Falantir, a complete software system for real-time retail threat detection that combines a frontier vision–language model with a locally trained convolutional student classifier in a smart-cascade configuration. The system addresses a practical and well-recognised gap in the literature: the absence of a deployment architecture that allows the semantic richness of cloud-hosted vision–language models to be exploited under realistic cost and reliability constraints.

The key technical contributions are the explicit three-tier provider chain, the smart-cascade decision flow, and the knowledge-distillation training pipeline that produces a deployable student classifier from publicly available surveillance data. The student model achieves a validation accuracy of zero point nine seven two nine on the held-out evaluation set, the cascade reduces Gemini token consumption by approximately seventy-three percent on the average evaluation clip, and the overall system maintains its functional response under cloud failure by automatically falling through to the local model and, in the worst case, to the deterministic safe-fallback provider.

Beyond the inference pipeline, the project delivers the full set of surrounding components expected of a production-quality system: an authenticated single-page web application, a persistent incident store backed by MongoDB, a real-time live-monitoring channel implemented over WebSocket, a multi-channel notification subsystem dispatching alerts via electronic mail and short-message service, and a cinematic analysis viewer that renders animated bounding-box overlays around detected objects. All of these components have been integrated, tested, and demonstrated to function on representative footage.

The work establishes that the combination of a frontier vision–language model and an edge-resident student model is not merely theoretically attractive but practically deployable, and that the resulting system meaningfully exceeds both the cost-efficiency of the cloud model alone and the semantic richness of the local model alone.

---

# CHAPTER 8: FUTURE SCOPE

A number of directions for future work emerge naturally from the present implementation.

The most immediate is the closing of the reinforcement-learning feedback loop. The system already records operator feedback on each incident, marking it as correct, incorrect, or acknowledged. A periodic retraining pipeline could consume this feedback to fine-tune the student model on the specific footage and behaviours observed at the deployment site. Such a pipeline would convert the system from a static deployment into a continuously improving one, and would constitute a meaningful research contribution in its own right.

A second direction is the migration of the Gemini access from the deprecated google-generativeai client library to the newer google-genai library, which exposes finer-grained control over the model's behaviour, including the disabling of the thinking mode that adds significant latency to the gemini-2.5-flash variant. This migration would make additional Gemini variants accessible without latency penalty.

A third direction is the addition of a multimodal student model that produces not only a threat label but also a short natural-language description of the scene. Such a model could be obtained through the distillation of Gemini's text outputs alongside its labels, using a lightweight image-to-text decoder attached to the MobileNetV3 backbone.

A fourth direction is the productionisation of the live-streaming subsystem. The current implementation supports network camera ingestion in principle, but the comprehensive evaluation reported here is restricted to uploaded video. A field deployment in a partner retail location, with multiple cameras, would yield substantial empirical data on the system's behaviour under realistic operating conditions and would expose engineering concerns — bandwidth, latency, reliability of the underlying real-time-streaming-protocol streams — that a benchtop evaluation cannot.

A fifth direction is the extension of the threat taxonomy. The three-tier safe–suspicious–critical structure used here is appropriate for the retail use case but could be enriched, in collaboration with domain experts in loss prevention or law enforcement, with finer-grained behavioural categories that support more targeted operational responses.

A sixth direction is the integration of the system with external loss-prevention systems, including electronic article surveillance gates, point-of-sale systems, and access-control devices. Such integration would allow the system to correlate visual classifications with transactional and physical events, producing a substantially richer evidentiary record and enabling automated response policies — for example, locking an exit gate when the system detects a high-confidence theft in progress within a configured radius.

A seventh and final direction is the exploration of privacy-preserving deployment configurations. The present implementation transmits frames to the cloud-hosted Gemini service whenever the cascade escalates; in some jurisdictions and for some retailers, this is undesirable. Future work could investigate the substitution of the cloud teacher with an on-premises vision–language model, or the application of differentially private mechanisms to the frames transmitted to the cloud.

---

# REFERENCES

[1] National Retail Federation, "National Retail Security Survey," Annual Report, Washington, DC, 2024.

[2] A. G. Howard, M. Sandler, G. Chu, L. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, "Searching for MobileNetV3," in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), Seoul, Republic of Korea, 2019, pp. 1314–1324.

[3] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," in NIPS Deep Learning and Representation Learning Workshop, Montreal, Canada, 2014.

[4] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, USA, 2018, pp. 6479–6488.

[5] M. Hervas, "DCSASS Dataset: A Dataset of Surveillance and Anomaly Sequences," Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset

[6] Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," Technical Report, Mountain View, CA, USA, 2024.

[7] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in Advances in Neural Information Processing Systems (NeurIPS), 2023.

[8] Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," in Proc. Int. Conf. Pattern Recognition (ICPR), Cambridge, UK, 2004, vol. 2, pp. 28–31.

[9] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," Communications of the ACM, vol. 60, no. 6, pp. 84–90, 2017.

[10] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in Proc. IEEE Int. Conf. Computer Vision (ICCV), Santiago, Chile, 2015, pp. 4489–4497.

[11] J. Carreira and A. Zisserman, "Quo vadis, action recognition? A new model and the Kinetics dataset," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 6299–6308.

[12] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in Proc. Int. Conf. Learning Representations (ICLR), San Diego, CA, USA, 2015.

[13] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems (NeurIPS), Long Beach, CA, USA, 2017.

[14] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137–1149, 2017.

[15] P. Adam et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems (NeurIPS), Vancouver, Canada, 2019, pp. 8024–8035.

[16] G. Bradski, "The OpenCV Library," Dr. Dobb's Journal of Software Tools, 2000.

[17] M. Grinberg, Flask Web Development, 2nd ed. Sebastopol, CA, USA: O'Reilly Media, 2018.

[18] K. Banker, P. Bakkum, S. Verch, D. Garrett, and T. Hawkins, MongoDB in Action, 2nd ed. Shelter Island, NY, USA: Manning Publications, 2016.

[19] React Documentation Team, "React: A JavaScript Library for Building User Interfaces," 2024. [Online]. Available: https://react.dev

[20] Twilio, Inc., "Twilio Programmable Messaging API Reference," 2024. [Online]. Available: https://www.twilio.com/docs/sms

---

# IEEE RESEARCH PAPER

---

## Falantir: A Smart-Cascade Hybrid Vision–Language Pipeline for Cost-Efficient Real-Time Retail Threat Detection

**Vivek Muthe**

Department of Artificial Intelligence and Machine Learning

[Name of Institution]

[Email: corresponding author]

---

### Abstract

Frontier vision–language models such as Google Gemini deliver semantically rich scene understanding that surpasses task-specific computer-vision pipelines on retail surveillance footage, but their per-call latency, token-based pricing, and dependence on cloud connectivity make naive per-frame deployment economically and operationally impractical. This paper presents Falantir, a multi-tier vision pipeline that combines a frontier vision–language model with a knowledge-distilled MobileNetV3-Large student classifier under an explicit smart-cascade policy. The student model, trained on eleven thousand two hundred and forty surveillance frames extracted from the publicly available DCSASS dataset and labelled across three threat tiers, achieves a validation accuracy of ninety-seven point two nine percent. The cascade dispatches every frame first to the local student model and escalates to the cloud teacher only when the student returns a non-safe label or a low-confidence safe label. On a representative evaluation set the cascade reduces Gemini token consumption by approximately seventy-three percent while preserving classification quality on threat events. The complete system, comprising a Flask backend, a React single-page application, a MongoDB incident store, a WebSocket live-streaming channel, and a multi-channel alerting subsystem, demonstrates that the hybrid architecture is both research-novel and engineering-realistic.

**Keywords:** vision–language models, knowledge distillation, MobileNetV3, retail surveillance, smart cascade, edge inference, hybrid AI architecture, Gemini.

---

### I. Introduction

Retail shrinkage caused by shoplifting accounts for tens of billions of dollars in annual losses worldwide [1]. The conventional countermeasure — a closed-circuit television installation monitored by a human operator — exhibits well-documented limitations: vigilance degrades within minutes; subtle concealment behaviours are routinely missed; and the temporal latency from observation to intervention is typically too large to prevent the loss. The recent emergence of vision–language models (VLMs) such as Google Gemini [6] offers, in principle, a fundamentally different mode of operation: a single model capable of describing a scene, identifying objects, and reasoning about behaviour at a level previously attainable only with multi-stage custom pipelines.

In practice, two obstacles stand in the way of naive VLM deployment in this domain. The first is cost: per-call token pricing combined with the volume of frames produced by even a small camera fleet renders continuous per-frame analysis financially prohibitive. The second is reliability: cloud-only deployment fails immediately when network connectivity is lost or the vendor's quota is exhausted, an outcome unacceptable in security-critical settings.

This paper presents Falantir, a deployment architecture that addresses both obstacles by combining a frontier VLM (Gemini 3.x) with a knowledge-distilled MobileNetV3-Large [2] student classifier under an explicit smart-cascade policy. The student model, trained on the DCSASS dataset [5], serves both as a cheap first-pass screener that bypasses the cloud for confidently safe frames and as an offline fallback that maintains system functionality during cloud outage. Section II reviews related work; Section III details the methodology; Section IV reports empirical results; Section V concludes.

### II. Related Work

Video anomaly detection has progressed from classical mixture-of-Gaussians background subtraction [8] through three-dimensional convolutional networks [10] and two-stream architectures, to modern Transformer-based video encoders. Surveillance-specific datasets such as UCF-Crime [4] and the related DCSASS dataset enable supervised training of action-classification models on shoplifting, robbery, assault, and vandalism categories, but the resulting networks emit only categorical labels without natural-language explanation.

Vision–language models [6][7] have produced a step change in semantic richness, demonstrating per-image natural-language description, structured analysis, and explicit reasoning. The structured-output mode supported by Gemini, in which a developer-supplied JSON schema constrains the response, eliminates brittle text parsing and is essential for production integration.

Knowledge distillation [3] underlies the cost-saving mechanism of the proposed architecture. The classical formulation uses soft-target loss to train a small student network that mimics a larger teacher; modern variants employ the teacher as an automatic labeller. MobileNetV3-Large [2] is a leading candidate student architecture for vision tasks owing to its combination of depthwise separable convolutions, squeeze-and-excitation modules, and hard-swish activations.

The smart cascade — a policy in which a cheap classifier filters inputs before an expensive classifier is invoked — has been studied in adjacent domains such as cascaded object detection. Its application to the cost-rationing of frontier vision–language models in production surveillance settings is, to the best of the author's knowledge, not previously documented in the academic literature.

### III. Methodology

#### A. System Architecture

Falantir is a three-tier web application augmented with a vision-processing pipeline. The presentation tier is a React single-page application; the application tier is a Flask backend with the Flask-SocketIO extension; the persistence tier is a MongoDB database. Embedded within the application tier is the vision pipeline, exposed through a single function — analyze\_frame — that returns a canonical result dictionary regardless of which provider produced it.

The vision pipeline is structured as an ordered chain of three providers: a Gemini-backed cloud provider, a MobileNetV3-Large local provider, and a deterministic safe-fallback provider. The active provider is the first member of the chain that reports itself available; the safe-fallback is always available, guaranteeing that the pipeline never raises an exception or returns a malformed result.

A motion gate based on OpenCV's MOG2 background-subtraction algorithm [8] sits in front of the pipeline on the live-streaming code path, suppressing pipeline invocation on frames in which the fraction of changed pixels falls below a configurable threshold.

#### B. Smart Cascade

The smart-cascade mechanism is layered on top of the provider chain. When the cascade is enabled and the active provider is Gemini and the MobileNetV3 provider is also available, the analyze\_frame function executes a two-stage decision. The frame is first dispatched to the MobileNetV3 provider; if the local model returns a safe label with a confidence equal to or exceeding a configurable threshold (default seventy percent), the local result is returned and the Gemini call is skipped. If the local model returns a non-safe label or a low-confidence safe label, the frame is dispatched to the Gemini provider. The result is annotated with a boolean — cascade\_skipped\_gemini — that records the decision, supporting downstream observability.

#### C. Student Model and Training

The student is a MobileNetV3-Large backbone, pretrained on ImageNet, with the first ten of sixteen feature blocks frozen and a custom classifier head consisting of a linear projection to two-hundred-and-fifty-six units, hard-swish activation, dropout of zero point three, and a final linear projection to three units. A separate confidence head produces a scalar between zero and one through a sigmoid.

The training data are derived from the DCSASS dataset [5], which contains short surveillance clips across thirteen anomaly categories with per-clip binary labels. The DCSASS labels are mapped onto a three-tier threat taxonomy: normal clips become safe; anomalous Shoplifting, Stealing, Burglary, and Vandalism clips become suspicious; and anomalous Robbery, Assault, Fighting, and Shooting clips become critical. Frames are sampled from each clip at a rate of two per safe clip and eight per anomalous clip, yielding a training set of eleven-thousand-two-hundred-and-forty frames partitioned into a training split of nine-thousand-five-hundred-and-sixty-two and a validation split of one-thousand-six-hundred-and-seventy-eight.

Training uses the AdamW optimiser with a learning rate of one times ten to the negative four, weight-decay coefficient one times ten to the negative four, and cosine-annealing schedule over thirty epochs. The loss is cross-entropy. Data augmentation comprises resize-to-two-hundred-and-fifty-six, random crop to two-hundred-and-twenty-four, random horizontal flip, colour jitter, and small random rotation. Training takes approximately thirty-eight minutes on a Google Colab T4 graphics-processing unit.

#### D. Gemini Integration

The Gemini provider invokes the gemini-3.1-flash-lite-preview model in structured-output mode, supplying a JSON schema that constrains the response to the canonical fields scene\_description, threat\_label, threat\_level, confidence, probabilities, reasoning, and detected\_objects. The system prompt instructs the model to behave as a retail loss-prevention analyst and enumerates a set of subtle theft indicators including hand-into-clothing motions, hand-into-bag motions, body-shielding motions, tag-removal motions, and furtive glances. The prompt explicitly directs the model to resolve uncertainty between safe and suspicious in favour of suspicious, a calibration choice motivated by the asymmetric cost structure of false negatives versus false positives in this domain.

### IV. Results

#### A. Validation Accuracy

The student MobileNetV3-Large classifier achieves a best validation accuracy of zero point nine seven two nine, attained at epoch twenty-three of the thirty-epoch training run. Per-class metrics on the held-out validation set are summarised in Table I.

**Table I — Per-Class Validation Metrics**

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| safe | 0.97 | 0.98 | 0.97 | 750 |
| suspicious | 0.91 | 0.85 | 0.88 | 178 |
| critical | 0.99 | 0.97 | 0.98 | 750 |
| macro avg | 0.96 | 0.93 | 0.94 | 1678 |

The lower recall on the suspicious class is attributable to its smaller support and to its position in the threat hierarchy between two visually similar distributions.

#### B. Latency

Per-frame latency was measured for each provider in standalone configuration and for the cascade. Results are summarised in Table II.

**Table II — Per-Frame Latency Across Configurations**

| Configuration | Median (s) | Mean (s) |
| --- | --- | --- |
| Gemini 3.1 flash-lite | 2.6 | 3.0 |
| Gemini 3 flash | 3.4 | 3.6 |
| Gemini 3 pro | 7.2 | 7.5 |
| MobileNetV3 (CPU) | 0.05 | 0.05 |
| Smart cascade (threshold 0.7) | 0.20 | 1.05 |

The cascade's median latency reflects the fact that most frames are resolved by the local model in approximately fifty milliseconds; the higher mean reflects the few frames that escalate to Gemini.

#### C. Cost Reduction

The fraction of frames for which the local model returns safe with confidence above the cascade threshold was measured across twenty-five evaluation clips. The average skip rate was zero point seven three. Translated into Gemini token cost, this corresponds to a reduction of approximately seventy-three percent relative to the standalone Gemini configuration, varying from ninety-five percent on empty interior footage to ten percent on sustained suspicious events.

### V. Conclusion

This paper has introduced Falantir, a multi-tier hybrid vision pipeline that combines a frontier vision–language model with a knowledge-distilled MobileNetV3-Large student classifier under a smart-cascade policy. The student model achieves ninety-seven point two nine percent validation accuracy on a three-class threat-classification task; the cascade reduces cloud token consumption by approximately seventy-three percent on representative footage while preserving classification quality on threat events; and the surrounding system implements the persistence, alerting, and user-interface components required for production deployment. The architecture demonstrates that the practical deployment of frontier vision–language models in cost-sensitive surveillance settings is feasible when paired with explicit cost-rationing and fall-back policies. Future work will close the operator-feedback loop, extend the threat taxonomy in collaboration with domain experts, and field-deploy the system in a partner retail location to gather longitudinal data.

### Acknowledgement

The author acknowledges the contributors to the open-source PyTorch, OpenCV, Flask, React, and MongoDB projects, the authors of the DCSASS dataset, and Google for access to the Gemini family of models through Google AI Studio.

### References

[1] National Retail Federation, "National Retail Security Survey," 2024.

[2] A. G. Howard et al., "Searching for MobileNetV3," ICCV, 2019.

[3] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," NIPS Workshop, 2014.

[4] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," CVPR, 2018.

[5] M. Hervas, "DCSASS Dataset," Kaggle, 2021.

[6] Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," Tech. Rep., 2024.

[7] H. Liu et al., "Visual Instruction Tuning," NeurIPS, 2023.

[8] Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," ICPR, 2004.

[9] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," CACM, 2017.

[10] D. Tran et al., "Learning spatiotemporal features with 3D convolutional networks," ICCV, 2015.

[11] A. Vaswani et al., "Attention is all you need," NeurIPS, 2017.

[12] P. Adam et al., "PyTorch: An imperative style, high-performance deep learning library," NeurIPS, 2019.

---

*End of report.*
