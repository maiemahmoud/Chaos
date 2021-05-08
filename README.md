# README FILE
The matlab is used for feature extraction <br/>
Digitized ECG is the data set for the matlab code <br/>
The P,Q, R, S, T waves are the outputs of the matlab code (ECG features)<br/>
WEKA package is used to make raw ECG classification and QRS (after get it from the ECG features) classification <br/>
The clinical data are added to the ECG and QRS then make classification again (for each case) <br/>
The chaos theory is implemented on the QRSs and gives another data set which manipilated by python code for classification using four different algorithms(SVM, Neural Network, Naive Bayes, Dicision Tree <br/>
The results of the Dicision Tree is the best when applying chaos theory <br/>
