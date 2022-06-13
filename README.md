# Deep-Learning
Matlab Deep Learning
https://www.kaggle.com/datasets/gpiosenka/good-guysbad-guys-image-data-set veri seti kullanılarak eğitim gerçekleştirildi. Veri seti kendi içerisinde eğitim test olarak ayrıldığı
için aynı şekilde kullanıldı. 
Matlab üzerinden eğitim gerçekleştirildi. Aşağıda dosya okutma şekilleri, optimizasyon ve projede kullanılan katmanlar yer almaktadır.


trainingSetup = load("C:\Users\kubra\Desktop\MatProje\trainingSetup_2022_04_23__14_36_06.mat");
imdsTrain = imageDatastore("C:\Users\kubra\Desktop\MatProje\train","IncludeSubfolders",true,"LabelSource","foldernames");
imdsValidation = imageDatastore("C:\Users\kubra\Desktop\MatProje\valid","IncludeSubfolders",true,"LabelSource","foldernames");
imdsTest = imageDatastore("C:\Users\kubra\Desktop\MatProje\test","IncludeSubfolders",true,"LabelSource","foldernames");
% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([48 48 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([48 48 3],imdsValidation);
augimdsTest = augmentedImageDatastore([48 48 3],imdsTest);

opts = trainingOptions("adam",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",10,...
    "MiniBatchSize",32,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);
layers = [
    imageInputLayer([48 48 3],"Name","imageinput")
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([5 5],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([5 5],"Name","maxpool_3","Padding","same")
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
[net, traininfo] = trainNetwork(augimdsTrain,layers,opts);
analyzeNetwork(net)
net = trainNetwork(augimdsTest,layers,opts);
[YPred,scores] = classify(net,augimdsTest);
accuracy = mean(YPred == imdsTest.Labels)
confusionchart(YPred,imdsTest.Labels)

save net

%Kontrol amacli
figure
idx = randperm(length(augimdsTest.Files),5);
for i = 1:5
    subplot(5,5,i);
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end


I = imread("C:\Users\kubra\Desktop\MatProje\test\savory\001.jpg");
I2= imresize(I,[48,48],'nearest');
[Pred,scores] = classify(net,I2);
scores = max(double(scores*100));
imshow(I);
title(join([string(Pred),'' ,scores ,'%']))






