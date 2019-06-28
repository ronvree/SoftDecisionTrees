# SoftDecisionTrees
Code used during our REDI research project. Includes a Soft Decision Tree implementation as described in 'Distilling a Neural Network Into a Soft Decision Tree' by Nicholas Frosst and Geoffrey Hinton of the Google Brain Team, together with variants on this model as described in our research paper. Everything is implemented in PyTorch.

TLDR; all files that start with 'train_' can be run to train/test the corresponding Soft Decision Tree models.

Description of individual python files:
data.py - Used to load the MNIST and SoftMNIST datasets. 
data_soft_targets.py - Defines the SoftMNIST dataset. 
lenet.py - LeNet5 implementation used to generate soft labels. 
sdt_frosst_hinton_incomplete.py - A working but incomplete implementation of Soft Decision Trees as described by Frosst and                                     Hinton. Included for code readability purposes, as it is a simplified Soft Decision Tree.                                     The complete version adds a layer of complexity. See code comments for a detailed                                             description.
sdt_frosst_hinton.py - An implementation of Soft Decision Trees as described in 'Distilling a Neural Network Into a Soft                            Decision Tree' by Nicholas Frosst and Geoffrey Hinton.
sdt_dropout.py - A Soft Decision Tree implementation using dropout regularization.
sdt_alternative.py - Extends the Soft Decision Tree model used by Frosst and Hinton to support alternative training                                procedures. See our paper/code for a detailed description of this alternative procedure.
treevis.py - Code used to generate .dot visualizations of the Soft Decision Trees. Note: Creating visualizations break the                decision trees. This bug was found too late to fix before the hand-in deadline. However, it has not affected the              results described in our research paper, as we did not visualize during these runs.
train_lenet.py - code used to train LeNet 5.
train_sdt_frosst_hinton.py - code used to train the original Soft Decision Tree implementation. 
train_sdt_frosst_hinton_incomplete.py - code used to train the incomplete version of the original Soft Decision Tree                                                 implementation.
train_sdt_dropout.py - code used to train the Soft Decision Tree with dropout regularization.
train_sdt_alternative.py - code with alternative training procedure for the tree implemented in sdt_alternative.py
