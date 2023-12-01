*In this blog, we will understand the idea behind Parameter Efficient Fine Tuning (PEFT), and explore LoRA and QLoRA, Two of the most important PEFT methods. We will understnad how PEFT can be used to fine tune the model for domain specific tasks, at the lowest cost and minimal infrastrcuture.在这篇博客中，我们将了解参数高效微调 （PEFT） 背后的思想，并探索 LoRA 和 QLoRA，这两种最重要的 PEFT 方法。我们将了解如何使用 PEFT 以最低的成本和最小的基础设施来微调特定领域任务的模型。*

# Motivation 赋予动机

In the ever-evolving world of AI and Natural Language Processing (NLP), Large Language Models and Generative AI have become powerful tools for various applications. Achieving the desired results from these models involves different approaches that can be broadly classified into three categories: Prompt Engineering, Fine-Tuning, and Creating a new model. As we progress from one level to another, the requirements in terms of resources and costs increase significantly.在不断发展的人工智能和自然语言处理 （NLP） 世界中，大型语言模型和生成式人工智能已成为各种应用的强大工具。从这些模型中实现预期结果涉及不同的方法，这些方法大致可分为三类：提示工程、微调和创建新模型。随着我们从一个级别发展到另一个级别，在资源和成本方面的要求显着增加。

In this blog post, we’ll explore these approaches and focus on an efficient technique known as Parameter Efficient Fine-Tuning (PEFT) that allows us to fine-tune models with minimal infrastrcture while maintaining high performance.在这篇博文中，我们将探讨这些方法，并重点介绍一种称为参数高效微调 （PEFT） 的有效技术，该技术使我们能够在保持高性能的同时，以最小的基础设施微调模型。

## Prompt Engineering with Existing Models使用现有模型进行快速工程设计

At the basic level, achieving expected outcomes from Large Language Models involves careful prompt engineering. This process involves crafting suitable prompts and inputs to elicit the desired responses from the model. Prompt Engineering is an essential technique for various use cases, especially when general responses suffice.在基本层面上，实现大型语言模型的预期结果涉及仔细的提示工程。此过程涉及制作合适的提示和输入，以从模型中引出所需的响应。提示工程是各种用例的基本技术，尤其是当一般响应就足够时。

## Creating a New Model 创建新模型

At the highest level, Creating a new model involves training a model from scratch, specifically tailored for a particular task or domain. This approach provides the highest level of customization, but it demands substantial computational power, extensive data, and time.在最高级别上，创建新模型涉及从头开始训练模型，专门针对特定任务或领域量身定制。这种方法提供了最高级别的定制，但它需要大量的计算能力、大量的数据和时间。

## Fine Tuning Existing Models微调现有模型

When dealing with domain-specific use cases that require model adaptations, Fine Tuning becomes essential. Fine-Tuning allows us to leverage existing pre-trained foundation models and adapt them to specific tasks or domains. By training the model on domain-specific data, we can tailor it to perform well on targeted tasks.在处理需要模型调整的特定领域用例时，微调变得至关重要。微调使我们能够利用现有的预训练基础模型，并使其适应特定的任务或领域。通过在特定领域的数据上训练模型，我们可以对其进行定制，使其在目标任务上表现良好。

However, this process can be resource-intensive and costly, as we will be modifying all the milions of paramters, as part of training. Fine tuning the model requires a lot of training data, huge infrastructure and effort.然而，这个过程可能是资源密集型和昂贵的，因为作为培训的一部分，我们将修改所有参数。微调模型需要大量的训练数据、庞大的基础设施和工作量。

=In the process of full fine-tuning of LLMs, there is a risk of = =*catastrophic forgetting*= =, where previously acquired knowledge from pretraining is lost.=

Applying complete fine-tuning to a single model for different domain-specific tasks often results in creating large models tailored to specific tasks, lacking modularity. What we require is a modular approach that avoids altering all parameters, while demanding fewer infrastructure resources and less data.针对不同的特定领域任务对单个模型进行完全微调通常会导致创建针对特定任务的大型模型，缺乏模块化。我们需要的是一种模块化方法，可以避免更改所有参数，同时需要更少的基础设施资源和数据。

There are various techniques such as Parameter Efficient Fine Tuning (PEFT), which provide a way to perform modular, fine-tuning with optimal resources and cost.有多种技术，例如参数高效微调 （PEFT），它们提供了一种以最佳资源和成本执行模块化微调的方法。

# Parameter Efficient Fine Tuning (PEFT)参数高效微调 （PEFT）

PEFT is a technique designed to fine-tune models while minimizing the need for extensive resources and cost. PEFT is a great choice when dealing with domain-specific tasks that necessitate model adaptation. By employing PEFT, we can strike a balance between retaining valuable knowledge from the pre-trained model and adapting it effectively to the target task with fewer parameters. There are various ways of achieving Parameter efficient fine-tuning. Low Rank Parameter or LoRA & QLoRA are most widely used and effective.PEFT 是一种旨在微调模型的技术，同时最大限度地减少对大量资源和成本的需求。在处理需要模型调整的特定于领域的任务时，PEFT 是一个不错的选择。通过使用 PEFT，我们可以在保留预训练模型中的宝贵知识与以更少的参数有效地适应目标任务之间取得平衡。有多种方法可以实现参数的高效微调。低秩参数或LoRA和QLoRA是最广泛和有效的。

## Low-Rank Parameters 低秩参数

This is one of the most widely used methods, where a set of parameters are modularly added to the network, with lower dimensional space. Instead of modifying the whole network, only these modular low-rank network is modified, to achieve the results.这是使用最广泛的方法之一，其中一组参数以模块化方式添加到网络中，具有较低的维空间。而不是修改整个网络，只修改这些模块化的低秩网络，以达到效果。

Let's deep dive into one of the most popular techniques called LoRA & QLoRA让我们深入了解最流行的技术之一，称为LoRA和QLoRA

## Low-Rank Adaptation (LoRA)低秩适应 （LoRA）

Low-Rank Adaptation provides the modular approach towards to fine-tuning a model for domains specific tasks and provides the capability of transfer learning. LoRA technique can be implemented with fewer resources and are memory efficient.低秩自适应提供了模块化方法，用于针对特定领域任务微调模型，并提供迁移学习功能。LoRA 技术可以用更少的资源实现，并且内存效率高.

In the following picture you can see the dimension/rank decomposition, that reduces the memory footprint considerably.在下图中，您可以看到维度/列分解，这大大减少了内存占用。

![](https://miro.medium.com/v2/resize:fit:700/1*OFO_oWGT2AnFn9o9mvmMvQ.png)

We will be aplying this by augmenting a LoRA adapter to the exisiting feed forward networks. We will be freezing the original feed forward networks, and will be using the LoRA network for training. Refer to the picture below for more details.我们将通过将 LoRA 适配器添加到现有的前馈网络来实现这一目标。我们将冻结原始的前馈网络，并将使用 LoRA 网络进行训练。有关详细信息，请参阅下图。

![](https://miro.medium.com/v2/resize:fit:700/1*rOW5plKBuMlGgpD0SO8nZA.png)

1. LoRA can be implemented as an adapter designed to enhance and expand the existing neural network layers. It introduces an additional layer of trainable parameters (weights) while maintaining the original parameters in a frozen state. These trainable parameters possess a substantially reduced rank (dimension) compared to the dimensions of the original network. This is the mechanism through which LoRa simplifies and expedites the process of adapting the original models for domain-specific tasks. Now, let’s take a closer look at the components within the LORA adapter network.LoRA 可以作为适配器实现，旨在增强和扩展现有的神经网络层.它引入了一个额外的可训练参数（权重）层，同时将原始参数保持在冻结状态。与原始网络的维度相比，这些可训练参数的秩（维度）大大降低。这是 LoRa 简化和加快将原始模型调整为特定领域任务的过程的机制。现在，让我们仔细看看 LORA 适配器网络中的组件。
2. The pre-trained parameters of the original model (`<strong class="mj gf"><em class="nd">W</em></strong>`) are frozen. During training, these weights will not be modified.原始模型 （ W ） 的预训练参数被冻结。在训练期间，这些权重不会被修改。
3. A new set of parameters is concurrently added to the networks `WA` and `WB`. These networks utilize low-rank weight vectors, where the dimensions of these vectors are represented as `dxr` and `rxd`. Here, ‘d’ stands for the dimension of the original frozen network parameters vector, while ‘r’ signifies the chosen low-rank or lower dimension. The value of ‘r’ is always smaller, and the smaller the ‘r’, the more expedited and simplified the model training process becomes. Determining the appropriate value for ‘r’ is a pivotal decision in LoRA. Opting for a lower value results in faster and more cost-effective model training, though it may not yield optimal results. Conversely, selecting a higher value for ‘r’ extends the training time and cost, but enhances the model’s capability to handle more complex tasks.一组新的参数同时添加到网络 WA 和 WB 中。这些网络使用低秩权重向量，其中这些向量的维度表示为 dxr 和 rxd 。这里，“d”代表原始冻结网络参数向量的维度，而“r”表示选择的低秩或较低维度。“r”的值总是越小，“r”越小，模型训练过程就越快速和简化。确定“r”的适当值是 LoRA 中的关键决定。选择较低的值会导致更快、更具成本效益的模型训练，尽管它可能不会产生最佳结果。相反，为“r”选择较高的值会延长训练时间和成本，但会增强模型处理更复杂任务的能力。
4. The results of the original network and the low-rank network are computed with a dot product, which results in a weight matrix of n dimension, which is used to generate the result.用点积计算原始网络和低秩网络的结果，得到n维的权重矩阵，用于生成结果。
5. This result is then compared with the expected results (during training) to calculate the loss function and WA and WB weights are adjusted based on the loss function as part of backpropagation like standard neural networks.然后将该结果与预期结果（在训练期间）进行比较，以计算损失函数，并根据损失函数调整 WA 和 WB 权重，作为反向传播的一部分，如标准神经网络。

Let’s explore how this approach contributes to the reduction of the memory footprint and minimizes infrastructure requirements. Consider a scenario where we have a 512x512 parameter matrix within the feed-forward network, amounting to a total of 262,144 parameters that need to undergo training. If we choose to freeze these parameters during the training process and introduce a LoRA adapter with a rank of 2, the outcome is as follows: WA will have 512*2 parameters and WB will also have 512*2 parameters, summing up to a total of 2,048 parameters. These are the specific parameters that undergo training with domain-specific data. This represents a significant enhancement in computational efficiency, substantially reducing the number of computations required during the backpropagation process. This mechanism is pivotal in achieving accelerated training.让我们探讨一下这种方法如何有助于减少内存占用并最大程度地降低基础结构要求。假设前馈网络中有一个 512x512 的参数矩阵，总共有 262,144 个参数需要接受训练。如果我们选择在训练过程中冻结这些参数，并引入一个秩为 2 的 LoRA 适配器，结果如下：WA 将有 512*2 个参数，WB 也将有 512*2 个参数，总共有 2,048 个参数。这些是使用特定于域的数据进行训练的特定参数。这代表了计算效率的显著提高，大大减少了反向传播过程中所需的计算次数。这种机制对于实现加速训练至关重要。

The most advantageous aspect of this approach is that the trained LoRA adapter can be preserved independently and employed as distinct modules. By constructing domain-specific modules in this manner, we effectively achieve a high level of modularity. Additionally, by refraining from altering the original weights, we successfully circumvent the issue of catastrophic forgetting.这种方法最有利的方面是，经过训练的 LoRA 适配器可以独立保存并用作不同的模块。通过以这种方式构建特定领域的模块，我们有效地实现了高度的模块化。此外，通过避免改变原始权重，我们成功地规避了灾难性遗忘的问题。

Now, let’s delve into further enhancements that can be implemented atop LoRA, particularly through the utilization of QLoRA, in order to elevate the optimization to the next level.现在， 让我们深入研究可以在 LoRA 上实现的进一步增强功能， 特别是通过使用 QLoRA， 以便将优化提升到一个新的水平.

## Quantized Low-Ranking Adaptation (QLoRA)量化低秩适应 （QLoRA）

QLoRA extends LoRA to enhance efficiency by quantizing weight values of the original network, from high-resolution data types, such as Float32, to lower-resolution data types like int4. This leads to reduced memory demands and faster calculations.QLoRA 扩展了 LoRA 以通过量化原始网络的权重值来提高效率，从高分辨率数据类型（如 Float32）到低分辨率数据类型（如 int4）。这样可以减少内存需求并加快计算速度。

There are 3 Key optimizations that QLoRA brings on top of LoRA, which makes QLoRA one of the best PEFT methods.QLoRA 在 LoRA 之上带来了 3 个关键优化，这使得 QLoRA 成为最好的 PEFT 方法之一。

**4-bit NF4 Quantization 4 位 NF4 量化**

4-bit NormalFloat4 is an optimized data type that can be used to store weights, which brings down the memory footprint considerably. 4-bit NormalFloat4 quantization is a 3-step process4 位 NormalFloat4 是一种优化的数据类型，可用于存储权重，从而大大减少了内存占用。4 位 NormalFloat4 量化是一个 3 步过程

 **Normalization & Quantization** : As part of normalization and quantization steps, the weights are adjusted to a zero mean, and a constant unit variance. A 4-bit data type can only store 16 numbers. As part of normalization the weights are mapped to these 16 numbers, zero-centered distributed, and instead of storing the weights, the nearest position is stored. Here is an example归一化和量化：作为归一化和量化步骤的一部分，权重被调整为零平均值和恒定的单位方差。4 位数据类型只能存储 16 个数字。作为归一化的一部分，权重被映射到这 16 个数字，以零为中心分布，而不是存储权重，而是存储最近的位置。下面是一个示例

Let's say we have a FP32 weight, with a value of 0.2121. a 4-bit split between -1 to 1 will be the following number positions.假设我们有一个 FP32 权重，值为 0.2121。-1 到 1 之间的 4 位拆分将是以下数字位置。

![](https://miro.medium.com/v2/resize:fit:700/1*BGbuLFqFPlgX0609Vd4oKg.png)

0.2121 is closest to 0.1997, which is the 10th position. Instead of saving the FP32 of 0.2121, we store 10.0.2121 最接近 0.1997，即第 10 位。我们没有保存 0.2121 的 FP32，而是存储 10。

The typical formula 典型公式

```
int4Tensor = roundedValue(totalNumberOfPositions/absmax(inputXTensor)) 
                    * FP32WeightsTensor

In the above example 
totalNumberOfPositions = 16
```

The value `totalNumberOfPositions/absmax(inputXTensor)` is called the quantization constant该值 totalNumberOfPositions/absmax(inputXTensor) 称为量化常数

Obviously, there is a loss of data when we normalize and quantize, as we move from FP32, which is a high-resolution data type to a low-resolution data type. The loss is not huge, as long as there are no outliers in the input tensor, which might affect the absmax () and eventually upset the distribution. To avoid that issue, we generally quantize the weights independently by smaller blocks, which will normalize the outliers.显然，当我们从 FP32（高分辨率数据类型）移动到低分辨率数据类型时，当我们进行归一化和量化时，会丢失数据。只要输入张量中没有异常值，损失就不大，这可能会影响 absmax（） 并最终扰乱分布。为了避免这个问题，我们通常通过较小的块独立量化权重，这将使异常值归一化。

 **Dequantization** : To Dequantize the values, we do exactly the reverse.去量化：为了去量化值，我们做恰恰相反的事情。

```
dequantizedTensor = int4Tensor
                     /roundedValue(totalNumberOfPositions/absmax(inputXTensor))

In the above example 
totalNumberOfPositions = 16
```

The 4-bit NormalFloat quantization is applied to the weights of the original model, the LoRA adapter weights will be FP32, as all of the training will happen on these weights. Once all the training is done, the original weights will be de-quantized.4 位 NormalFloat 量化应用于原始模型的权重，LoRA 适配器权重将为 FP32，因为所有训练都将在这些权重上进行。完成所有训练后，原始权重将被去量化。

![](https://miro.medium.com/v2/resize:fit:700/1*zYSQyksAi4u2OoZ0sM5-4w.png)

**Double Quantization 双重量化**

Double quantization, further reduces the memory footprint, by quantizing, quantization constants. In the previous 4-bit FP4 quantization step, we calculated the quantization constant. Even that can be quantized for better efficiency, and that is what we do in Double Quantization.双重量化，通过量化、量化常数，进一步减少内存占用。在前面的 4 位 FP4 量化步骤中，我们计算了量化常数。即使这样也可以被量化以获得更高的效率，这就是我们在双重量化中所做的。

Since the quantization is done in blocks, to avoid outliers, typically 64 weights in 1 block, we will have 1 quantization constant. These quantization constants can be quantized further, to reduce the memory footprint.由于量化是在块中完成的，为了避免异常值，通常 1 个块中有 64 个权重，我们将有 1 个量化常数。这些量化常数可以进一步量化，以减少内存占用。

Let's say we have grouped 64 parameters/weights per block, and each quantization constant takes 32 bits, as it is FP32. It adds a 0.5 bit per parameter on average, which means we are talking of at least 500,000 bits for a typical 1Mil parameter model.假设我们对每个块的 64 个参数/权重进行了分组，每个量化常数需要 32 位，因为它是 FP32。它平均为每个参数增加 0.5 位，这意味着对于典型的 1Mil 参数模型，我们谈论的至少是 500,000 位。

With Double quantization, we apply quantization on these quantization constants, which will further optimize our memory usage. We can take a group of 256 quantization values, and apply 8-bit quantization. we can achieve approximately 0.127 bits per parameter, which brings down the value to 125,000 bits for the 1Mil parameter model.通过双重量化，我们对这些量化常数应用量化，这将进一步优化我们的内存使用。我们可以采用一组 256 个量化值，并应用 8 位量化。我们可以实现每个参数大约 0.127 位，这使得 1Mil 参数模型的值降至 125,000 位。

> Here is the calculation: We have 64 weights in 256 blocks which are 32 bits which is 32/(64*256) which is 0.001953125计算如下：我们在 256 个块中有 64 个权重，它们是 32 位，即 32/（64*256），即 0.001953125
>
> We have 8bits for 64 weights which is 8/64 0.125我们有 8 位用于 64 个权重，即 8/64 0.125
>
> If we add it up 0.125+0.001953125 which is 0.127 approximately如果我们将其加起来 0.125+0.001953125，大约是 0.127

**Unified Memory Paging 统一内存分页**

Coupled with the above techniques, QLoRA also utilizes the nVidia unified memory feature, which allows GPU->CPU seamless page transfers, when GPU runs out of memory, thus managing the sudden memory spikes in GPU, and helping memory overflow/overrun issues.结合上述技术，QLoRA 还利用了 nVidia 统一内存功能，当 GPU 内存不足时，该功能允许 GPU >CPU 无缝页面传输，从而管理 GPU 中突然出现的内存峰值，并帮助解决内存溢出/溢出问题。

LoRA and QLoRA are two of the most emerging and widely used techniques for Parameter Efficient Fine tuning.LoRA 和 QLoRA 是参数高效微调的两种最新兴和最广泛使用的技术。

In the next part, we will implement QLoRA, until then, have fun with LLMs在下一部分中，我们将实现 QLoRA，在此之前，请享受 LLM 的乐趣

Hope this was useful, leave your comments and feedback...希望这对您有用，留下您的评论和反馈......

Bye for now... 再见。。。

# References 引用

* [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
* [https://arxiv.org/abs/2304.01933](https://arxiv.org/abs/2304.01933)
