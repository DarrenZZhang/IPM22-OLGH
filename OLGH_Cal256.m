clear all;
nbits_set = [8,16,32,64,96,128];
load('Caltech256_1kdim')
feat = double(feat);
nTest = 1000;% rng(100)
test_list = randsample(size(feat,1), nTest);
label = zeros(length(rgbImgList),1);
for i = 1:length(rgbImgList)
    str = rgbImgList{i};
    label(i) = str2double(str(1:3));
end
testdata = feat(test_list,:);
trfea = feat; clear feat;
trfea(test_list,:) = [];
traindata  = trfea;
testgnd = label(test_list);
label(test_list) = [];
traingnd = label;
WtrueTestTraining = bsxfun(@eq, testgnd, traingnd');

exp_data.traindata = double(traindata);
exp_data.testdata = double(testdata);

%% Anchor features
X = exp_data.traindata;
n_anchors = 1000; 
anchor = X(randsample( size(exp_data.traindata,1), n_anchors),:);
Dis = EuDist2(X, anchor, 0);
sigma = mean(min(Dis,[],2).^0.5);
Phi_testdata = exp(-sqdist_sdh(exp_data.testdata,anchor)/(2*sigma*sigma));
Phi_traindata = exp(-sqdist_sdh(exp_data.traindata,anchor)/(2*sigma*sigma));
X = [Phi_traindata ; Phi_testdata];
data_our.indexTrain=1:size(exp_data.traindata,1);
data_our.indexTest=size(exp_data.traindata,1)+1:size(exp_data.traindata,1) + size(exp_data.testdata,1);
data_our.X = normZeroMean(X);
data_our.X = normEqualVariance(X);
data_our.label = double([traingnd;testgnd]);


for ii = 1:length(nbits_set)
    nbits = nbits_set(ii);
    opt.lambda    = 50;
    opt.beta      = .1;
    opt.gamma     = 10;
    opt.eta       = .1;

    opt.k         = 5;
    opt.Iter_num  = 5;
    t1 = tic;
    [B_trn, B_tst] = OLGH(data_our, opt, nbits);
    trtime = toc(t1);

    B1 = compactbit(B_trn);
    B2 = compactbit(B_tst);
    %% Evaluation
    DHamm = hammingDist(B2, B1);
    [~, orderH] = sort(DHamm, 2);
    MAP = calcMAP(orderH, WtrueTestTraining);
    fprintf('Bits: %d, MAP: %.4f...   \n', nbits, MAP);
end