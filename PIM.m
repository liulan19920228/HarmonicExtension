%function PIM_image = PIM()

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
    subsample_rate = 0.01;
    S = floor(subsample_rate*P);
    mu = P/S;
    
    %construct the labelled subsample set     
%   load('PIM_IDX.mat');
%   load('PIM_DIS.mat');
%    load('subsample_one_iter.mat');
    subsample = randperm(P,S);
    save('subsample_one_iter.mat','subsample');

    
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
    [IDX, DIS] = knnsearch(patch,patch,'k',num_neigh+1,'NSMethod','kdtree','Distance','euclidean');
 %   save('PIM_subsample.mat','subsample');
 %   save('PIM_IDX.mat','IDX')
 %   save('PIM_DIS.mat','DIS')
    
    
    %construct the weight matrix
    weight = exp(-bsxfun(@rdivide, DIS(:,2:51), DIS(:,21)).^2);
    
    %construct the sparse linear system, takes 618s 
    fprintf('construct A and b\n')
    A = sparse(P, P);
    b = zeros(P,1);
    A(1:P+1:end) = sum(weight,2);%update the diagonal
    
    %temp =1:P;
    %A(bsxfun(@plus,(IDX(:,2:51)-1)*P,temp')) = -weight;
    for i=1:P
        [labeled, labeled_pos] = intersect(IDX(i,2:51),subsample);                
        A(i,IDX(i,2:51)) = -weight(i,:);
        A(i,labeled) = A(i,labeled) + mu*weight(i,labeled_pos);
        b(i) = mu*weight(i,labeled_pos)*image_new(labeled);
    end
    
    %save('PIM_A.mat','A');
    %save('PIM_b.mat','b');
    %load('PIM_A.mat');
    %load('PIM_b.mat');
    
    fprintf('Solving the linear system\n')
    sol = A\b;%takes 241s
    u = full(sol);    
    PSNR = -20*log10(norm(image_new-u)/255^2)  
    save('PSNR_One_Iter_PIM.mat','PSNR');
    PI = reshape(uint8(u), nrow, ncol);
    h = imshow(PI);
    saveas(h, sprintf('One_Iter_PIM%d.png',1));
    close

%end

