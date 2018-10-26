%function Iter_image = Iter()
tic
    %paramaters declaim
    fprintf('parameters declaim...\n')
    %imshow('lenna_grayscale.tif');%figure1: original figure
    image = double(imread('lenna_grayscale.tif'));
    nrow = size(image,1);
    ncol = size(image,2);
    num_neigh = 50;
    P = nrow*ncol;
    image_new = reshape(image,P,1);%reorder col by col
    patch_size = 11;%must be odd number 
    window = floor(patch_size/2);
    u = zeros(P,1);
    subsample_rate = 0.1;
    S = floor(subsample_rate*P);
    PSNR = zeros(12,1);
    
    %subsample = randperm(P,S);
    %save('subsample.mat','subsample');
    load('subsample.mat');
    pixel = zeros(P,1);
    pixel(subsample) = image_new(subsample);
    PSNR(1)=-20*log10(norm(image_new-pixel)/255^2);
    figure
    h=imshow(reshape(uint8(pixel),nrow,ncol));%figure2: only with subsample
    saveas(h,sprintf('FIG_GLM%d.png',1));
    close

    
    pos = setdiff(1:P, subsample);
    inverse = zeros(P,1);
    inverse(pos) = 1:P-S;  
    f_omega_inf = max(abs(image_new(subsample)));
    lambda1 = 3*f_omega_inf/nrow;
    lambda2 = 3*f_omega_inf/ncol;
    f_mu = mean(image_new(subsample));
    f_sigma = std(image_new(subsample));
    image = normrnd(f_mu,f_sigma,P,1);
    image = max(min(image,255),0);
    image(subsample) = image_new(subsample);
    PSNR(2)=-20*log10(norm(image_new-image)/255^2);%
    image = reshape(image, nrow,ncol);%figure3: subsample + random
    figure
    h=imshow(uint8(image));%figure3
    saveas(h,sprintf('FIG_GLM%d.png',2));
    close

    
    for iter = 1:20 
        iter
        expand_image = [flipud(image(1:window,:)); image; flipud(image(nrow-window+1:nrow,:))];
        expand_image = [fliplr(expand_image(:,1:window)),expand_image,fliplr(expand_image(:,ncol-window+1:ncol))];     
        patch = zeros(P,patch_size^2+2);
        for j = 1:ncol
            for i = 1:nrow
                index = i+(j-1)*nrow;
                rLo = i;
                rHi = i + 2*window;
                cLo = j;
                cHi = j + 2*window;
                patch(index,:) = [reshape(expand_image(rLo:rHi, cLo:cHi),1,patch_size^2),lambda1*i,lambda2*j];
            end
        end
       
        [IDX, DIS] = knnsearch(patch,patch(pos,:),'k',num_neigh+1,'NSMethod','kdtree','Distance','euclidean');
        weight = exp(-bsxfun(@rdivide, DIS(:,2:51), DIS(:,21)).^2);
    
        A = sparse(P-S, P-S);
        b = zeros(P-S,1);
        A(1:P-S+1:end) = sum(weight,2);%update the diagonal
    
        for i=1:P-S
            [labeled, labeled_pos] = intersect(IDX(i,2:51),subsample);
            unlabeled_pos = setdiff(1:50, labeled_pos);
            A(i,inverse(IDX(i,unlabeled_pos+1)))=-weight(i,unlabeled_pos);
            b(i) = weight(i,labeled_pos)*image_new(labeled);
        end
       
        sol = A\b;%takes 238s
        u(pos) = full(sol);
        u(subsample) = image_new(subsample);
        PSNR(iter+2)=-20*log10(norm(image_new-u)/255^2)
        image = reshape(u, nrow, ncol);
        figure
        h=imshow(uint8(image));
        saveas(h,sprintf('FIG_GLM%d.png',iter+2));
        close
    end
    save('PSNR_GLM.mat','PSNR');
toc
