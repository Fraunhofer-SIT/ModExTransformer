# Transformer-based Extraction of Deep Image Models, EuroSP'22
This is the official repository for the paper **Transformer-based Extraction of Deep Image Models** presented at the IEEE 7th European Symposium on Security and Privacy (EuroS&P'22).

In our paper, we propose to use a transformer model, namely Facebook's DeiT, to copy deep image models. Hence, this repository is largely based on Facebook Research's [deit](https://github.com/facebookresearch/deit)-repository.
In addition, the defense mechanisms described in Section 6 of the paper were adopted from Tribhuvanesh Orekondy's [prediction-poisoning](https://github.com/tribhuvanesh/prediction-poisoning)-repo to test the robustness of the proposed attacker against SOTA defenses.

**Abstract:** Model extraction attacks pose a threat to the security of ML models and to the privacy of the data used for training. Previous research has shown that such attacks can be either monetarily motivated to gain an edge over competitors or maliciously in order to mount subsequent attacks on the extracted model. In this paper, recent advances in the field of transformers are exploited to propose an attack tailored to the task of image classification that allows stealing complex convolutional neural network models without any knowledge of their architecture. The attack was performed on a range of datasets and target architectures to evaluate the robustness of the proposed attack. With only 100k queries, we were able to recover up to 99.2% of the black-box target network's accuracy on the test set. We conclude that it is possible to effectively steal complex neural networks with relatively little expertise and conventional means -- even without knowledge of the target's architecture. Recently proposed defences have also been examined for their effectiveness in preventing the attack proposed in this paper.


## Dependencies

    pip install timm torch torchvision
    
If anything breaks, this environment has been tested:

    pip install -r requirements.txt

## Basic usage
All arguments can be provided both via the command line and via configuration files. 
For an overview of the available arguments, run 

    python main.py --help
    
or look [here](basic_config.ini). 

The general syntax for running the script is

    python main.py [-c [<config files>]+] [<other arguments>]*

All configuration files (in order from left to right) override the default values of the argument parser.
If you add other arguments on the command line, they will be used.

## Examples
In the folder [examples](examples) you can find some examples of how to use this script.

### Train target
To train a Resnet34 model on CIFAR10, run 

    python main.py -c basic_config.ini examples/target.ini


### Train attacker
To attack this Resnet34 target model with a DeiT-base model on SVHN, run 

    python main.py -c basic_config.ini examples/attacker.ini


### Attack defended targets
If you want to perform the same attack, but this time on the target defended with MAD, run

    python main.py -c basic_config.ini examples/attacker.ini examples/attacker_defended.ini
    
You can also perform the very same attack without augmentation via

    python main.py -c basic_config.ini examples/attacker.ini examples/attacker_defended.ini --no-augmentation
    
(Of course, you could as well add `augmentation = False` to _attacker_defended.ini_)


