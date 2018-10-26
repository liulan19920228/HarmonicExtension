%function GLM_image = GLM()

    %paramaters declaim
    fprintf('parameters declaim...\n')
    %imshow('lenna_grayscale.tif');
    image = double(imread('lenna_grayscale.tif'));
    nrow = size(image,1);
    ncol = size(image,2);
    num_neigh = 50;
    P = nrow*ncol;
    image_new = reshape(image,P,1);%reorder col by col
    patch_size = 9;%must be odd number 
    window = floor(patch_size/2);
    u = zeros(P,1);
    subsample_rate = 0.01;
    S = floor(subsample_rate*P);
    
    %construct the labelled subsample set     
%     load('IDX.mat');
%     load('DIS.mat');
    load('subsample_one_iter.mat');
    %subsample = randperm(P,S);
    pos = setdiff(1:P, subsample);
    inverse = zeros(P,1);
    inverse(pos) = 1:P-S;  
    
    %reflect the image in order to construct the patch
    expand_image = [flipud(image(1:window,:)); image; flipud(image(nrow-window+1:nrow,:))];
    expand_image = [fliplr(expand_image(:,1:window)),expand_image,fliplr(expand_image(:,ncol-window+1:ncol))];    
    
    fprintf('constructing the patching matrix...\n')   
    patch = zeros(P,patch_size^2);
    for j = 1:ncol
        for i = 1:nrow
            index = i+(j-1)*nrow;
            rLo = i;
            rHi = i + 2*window;
            cLo = j;
            cHi = j + 2*window;
            patch(index,:) = reshape(expand_image(rLo:rHi, cLo:cHi),1,patch_size^2);
        end
    end
       
    %construct the 50 nearest neighbors and distance for all points.
    [IDX, DIS] = knnsearch(patch,patch(pos,:),'k',num_neigh+1,'NSMethod','kdtree','Distance','euclidean');
%     save('subsample.mat','subsample');
%     save('IDX.mat','IDX')
%     save('DIS.mat','DIS')
    
    
    %construct the weight matrix
    weight = exp(-bsxfun(@rdivide, DIS(:,2:51), DIS(:,21)).^2);
    
    %construct the sparse linear system    
    fprintf('construct A and b\n')
    A = sparse(P-S, P-S);
    b = zeros(P-S,1);
    A(1:P-S+1:end) = sum(weight,2);%update the diagonal
    
    for i=1:P-S
        [labeled, labeled_pos] = intersect(IDX(i,2:51),subsample);
        unlabeled_pos = setdiff(1:50, labeled_pos);
        A(i,inverse(IDX(i,unlabeled_pos+1)))=-weight(i,unlabeled_pos);
        b(i) = weight(i,labeled_pos)*image_new(labeled);
    end
 

% This way to define A, b is a lot slower than above. 3682s~554s
%     tic   
%     for i = 1:P-S
%         for j = 2:51
%             neigh = IDX(i,j);
%             if inverse(neigh) == 0 %labeled
%                 b(i) = b(i) + weight(i,j-1)*image_new(neigh);
%             else %unlabeled
%                 l = inverse(neigh);
%                 A(i,l) = A(i,l) - weight(i,j-1);
%             end            
%         end
%         if rem(i,5000) == 0
%             i
%         end
%     end
%     toc
    
    %save('A.mat','A');
    %save('b.mat','b');
    %load('A.mat');
    %load('b.mat');
    
    fprintf('Solving the linear system\n')
    sol = A\b;%takes 238s
    u(pos) = full(sol);
    u(subsample) = image_new(subsample);
    PSNR = -20*log10(norm(image_new-u)/255^2)    
    save('PSNR_One_Iter_GLM.mat','PSNR');
    GL = reshape(uint8(u), nrow, ncol);
    h = imshow(GL);
    saveas(h, sprintf('One_Iter_GLM%d.png',1));
    close
%end

