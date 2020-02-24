%{ 
Author: Deepti Ghadiyaram

Description: Given an input RGB image, this method extracts the
FRIQUEE-Chroma features proposed in [1] in LAB color space.

Input: a MXN array of chroma map (computed from A and B color channels).

Output: Features extracted from the chroma map (as described in the section
Chroma Feature Maps in [1])

Dependencies: This method depends on the following methods:
    neighboringPairProductFeats.m
    sigmaMapFeats.m
    debiasedNormalizedFeats.m
    DoGFeat.m
    lapPyramidFeats.m

Reference:
[1] D. Ghadiyaram and A.C. Bovik, "Perceptual Quality Prediction on Authentically Distorted Images Using a
Bag of Features Approach," http://arxiv.org/abs/1609.04757
%}
function feat = friqueeChroma(imChroma)

    % Initializations
    scalenum=2;   
    imChroma1=imChroma;
    
    % The array that aggregates all the features from different feature
    % maps.
    %feat = [];
    feat = zeros(1,80);
    index = 0;
    % Some features are computed at multiple scales
    for itr_scale = 1:scalenum
        
        % 4 Neighborhood map features of Chroma map
       nFeat = neighboringPairProductFeats(imChroma);
       %feat = [feat nFeat];
       n = length(nFeat);
       feat(index+1:index+n) = nFeat;
       index = index + n;
       
        % Sigma features of Chroma map
        [sigFeat, sigmaMap] = sigmaMapFeats(imChroma);
        %feat = [feat sigFeat];
        n = length(sigFeat);
        feat(index+1:index+n) = sigFeat;
        index = index + n;
        
        % Chroma map's classical debiased and normalized map's features.
        dnFeat = debiasedNormalizedFeats(imChroma);
        %feat = [feat dnFeat];
        n = length(dnFeat);
        feat(index+1:index+n) = dnFeat;
        index = index + n;
        
        % Debiased and normalized map features of the sigmaMap of Chroma map.
        dnSigmaFeat = debiasedNormalizedFeats(sigmaMap);
        %feat = [feat dnSigmaFeat];
        n = length(dnSigmaFeat);
        feat(index+1:index+n) = dnSigmaFeat;
        index = index + n;

        % Scale down the image by 2 and repeat the feature extraction.
        imChroma = imresize(imChroma,0.5);
        
    end
    
    imChroma=imChroma1;
    
    % Features from the Difference of Gaussian (DoG) of Sigma Map 
    dFeat =DoGFeat(imChroma);
    %feat = [feat dFeat];
    n = length(dFeat);
    feat(index+1:index+n) = dFeat;
    index = index + n;
    
    % Laplacian of the Chroma Map
    lFeat = lapPyramidFeats(imChroma);
    %feat = [feat lFeat];
    n = length(lFeat);
    feat(index+1:index+n) = lFeat;
   
end