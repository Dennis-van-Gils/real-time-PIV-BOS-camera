%clr;

filelist = dir('B*.tif');
nFiles = length(filelist);

% ------------------------------------------------------------------------
%   Walk over all files
% ------------------------------------------------------------------------

for iFile = 1:1
  % ----------------------------------------------------------------------
  %   Read images and remove background
  % ----------------------------------------------------------------------
  fn = char(filelist(iFile).name);
  img = imread(fn);
  
  % Read double image and split into frames A and B
  [img_h, img_w] = size(img);
  A = img(1:img_h/2, :);
  B = img(img_h/2+1:end, :);
  img_h = img_h/2;
  
  % Background removal
  A = A - mean(mean(A));
  B = B - mean(mean(B));
  
  % Original full figure A
  if 1
  h1 = figure(1); clf
  set(h1, 'Position', [60, 60, 500, 500])
  imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
  hold on
  tmp = get(h1, 'Position');
  set(h1, 'Position', [2626, -240, tmp(3), tmp(4)])
  end
  % Original full figure B
  if 1
  h3 = figure(3); clf
  set(h3, 'Position', [60, 60, 500, 500])
  imshow(imadjust(B), 'InitialMagnification', 80, 'Colormap', bone(256));
  hold on
  tmp = get(h3, 'Position');
  set(h3, 'Position', [2626, -240, tmp(3), tmp(4)])
  end
  
  % Prepare figure 2 for showing correlation map
  h2 = figure(2); clf
  set(h2, 'Position', [780, 1117, 256, 256])
  
  % ----------------------------------------------------------------------
  %   Calculate IWs
  % ----------------------------------------------------------------------
  
  % Set the IW sizes for multigrid analysis and shifting window technique
  % Subsequent IW sizes should be the exact half of the prev IW size
  IW_SIZES = [128 64 32];
  %IW_SIZES = [32];
  nIW_SIZES = length(IW_SIZES);
  
  % Allocate memory for multigrid vector maps
  vector_maps = cell(nIW_SIZES, 1);
  
  for iIW_SIZE = 1:nIW_SIZES
    fLastIW_SIZE = (iIW_SIZE == nIW_SIZES);   % Are we at the last IW_SIZE?
  
    % Set interrogation window size (IW) and calculate number of IWs
    IW_SIZE = IW_SIZES(iIW_SIZE);
    nIW_x = floor(img_w/IW_SIZE);
    nIW_y = floor(img_h/IW_SIZE);

    % Allocate memory for displacement vectors
    % 3rd dim will consecutively store [x, y, dx, dy]
    vector_map = zeros(nIW_x, nIW_y, 4);
    
    % Retrieve previous displacement vector map
    if iIW_SIZE > 1
      prev_vector_map = vector_maps{iIW_SIZE - 1};
    end
    
    %for iIW_x = ceil(img_w/2/IW_SIZE):ceil(img_w/2/IW_SIZE)+2
    %  for iIW_y = ceil(img_h/2/IW_SIZE):ceil(img_h/2/IW_SIZE)+2
    for iIW_x = 1:nIW_x
      for iIW_y = 1:nIW_y
        %fprintf('\niIW_x = %i, iIW_y = %i\n', iIW_x, iIW_y)
        
        % ----------------------------------------------------------------
        %   Retrieve interrogation window frame A
        % ----------------------------------------------------------------
        IW_A_x1 = (iIW_x - 1) * IW_SIZE + 1;
        IW_A_y1 = (iIW_y - 1) * IW_SIZE + 1;
        IW_A_x2 = IW_A_x1 + IW_SIZE - 1;
        IW_A_y2 = IW_A_y1 + IW_SIZE - 1;

        IW_A = A(IW_A_y1:IW_A_y2, IW_A_x1:IW_A_x2);
        
        % ----------------------------------------------------------------
        %   Retrieve interrogation window frame B
        %   Apply window shifting technique
        % ----------------------------------------------------------------                
        %if iIW_x == 21 && iIW_y == 1
        %  1;
        %end
        
        if iIW_SIZE == 1
          % First IW size, no pre-shift available
          shift_x = 0;
          shift_y = 0;
        else
          % Pre-shift available
          
          % Calculate corresponding index of prev IW
          get_prev_iIW_x = floor((iIW_x - 1)/2) + 1;
          get_prev_iIW_x = min(get_prev_iIW_x, size(prev_vector_map, 1));
          get_prev_iIW_y = floor((iIW_y - 1)/2) + 1;
          get_prev_iIW_y = min(get_prev_iIW_y, size(prev_vector_map, 2));
          
          shift_x = prev_vector_map(get_prev_iIW_x, get_prev_iIW_y, 3);
          shift_y = prev_vector_map(get_prev_iIW_x, get_prev_iIW_y, 4);
          
          if isnan(shift_x); shift_x = 0; end
          if isnan(shift_y); shift_y = 0; end
        end
        
        %fprintf('shift_x = %.1f, shift_y = %.1f\n', shift_x, shift_y);
        
        IW_B_x1 = IW_A_x1 + shift_x;
        IW_B_y1 = IW_A_y1 + shift_y;
        IW_B_x2 = IW_A_x2 + shift_x;
        IW_B_y2 = IW_A_y2 + shift_y;
        
        % The IW should never be shifted outside of frame B
        IW_B_x1 = max(1, IW_B_x1); IW_B_x1 = min(img_w, IW_B_x1);
        IW_B_x2 = max(1, IW_B_x2); IW_B_x2 = min(img_w, IW_B_x2);
        IW_B_y1 = max(1, IW_B_y1); IW_B_y1 = min(img_h, IW_B_y1);
        IW_B_y2 = max(1, IW_B_y2); IW_B_y2 = min(img_h, IW_B_y2);
        
        IW_B = B(IW_B_y1:IW_B_y2, IW_B_x1:IW_B_x2);

        % Show interrogation windows on top of image
        if 0
          figure(h1)
          plot([IW_A_x1 IW_A_x1 IW_A_x2 IW_A_x2 IW_A_x1] + ...
               [-.5 -.5 .5 .5 -.5], ... % esthetiques
               [IW_A_y1 IW_A_y2 IW_A_y2 IW_A_y1 IW_A_y1] + ...
               [-.5 .5 .5 -.5 -.5], ... % esthetiques
               '-r', 'LineWidth', 2)
          figure(h3)
          plot([IW_B_x1 IW_B_x1 IW_B_x2 IW_B_x2 IW_B_x1] + ...,
               [-.5 -.5 .5 .5 -.5], ... % esthetiques
               [IW_B_y1 IW_B_y2 IW_B_y2 IW_B_y1 IW_B_y1] + ...
               [-.5 .5 .5 -.5 -.5], ... % esthetiques
               '-y', 'LineWidth', 2)
        end

        % Perform cross-correlation
        if max(max(IW_A)) == 0 || max(max(IW_B)) == 0
          C = nan;                    % Save computation time
        else
          C = xcorr2(single(IW_B), single(IW_A));
          C = C/max(max(C));          % Normalize
        end

        % Show correlation map of current IW
        if 0
          figure(h2); cla
          imshow(C*255, jet(256), 'InitialMagnification', 300)
          hold on
          plot([IW_SIZE   IW_SIZE], [1       2*IW_SIZE]  , '-k', 'LineWidth', 1.5)
          plot([1       2*IW_SIZE], [IW_SIZE IW_SIZE], '-k', 'LineWidth', 1.5)
          axis on
        end

        % Find maximum correlation peak
        if isnan(max(max(C)))        
          x = nan; y = nan; dx = nan; dy = nan;
        else
          [maxC, iMaxC] = max(C(:));
          [peak_y, peak_x] = ind2sub(size(C), iMaxC);
          
          % Perform sub-pixel algorithm, Gaussian 3 point fit
          % But only on the smallest IW size
          if fLastIW_SIZE
            C_x_min1 = max(1, peak_x - 1);
            C_x_add1 = min(size(C, 2), peak_x + 1);
            C_y_min1 = max(1, peak_y - 1);
            C_y_add1 = min(size(C, 1), peak_y + 1);
            peak_x_G = peak_x + 1/2 * ...
              (log(C(peak_y, C_x_min1)) - log(C(peak_y, C_x_add1)) / ...
              (log(C(peak_y, C_x_min1)) + log(C(peak_y, C_x_add1)) - 2* ...
               log(C(peak_y, peak_x))));
            peak_y_G = peak_y + 1/2 * ...
              (log(C(C_y_min1, peak_x)) - log(C(C_y_add1, peak_x)) / ...
              (log(C(C_y_min1, peak_x)) + log(C(C_y_add1, peak_x)) - 2* ...
               log(C(peak_y, peak_x))));
            peak_x = peak_x_G;
            peak_y = peak_y_G;
          end

          % Store displacement vector (x, y, dx, dy)
          x  = IW_A_x1 + floor(IW_SIZE/2);
          y  = IW_A_y1 + floor(IW_SIZE/2);
          dx = peak_x - IW_SIZE + shift_x;
          dy = peak_y - IW_SIZE + shift_y;
        end

        % Store result in vector map
        vector_map(iIW_x, iIW_y, :) = [x, y, dx, dy];
        
        % Store vector map in vector maps
        vector_maps{iIW_SIZE} = vector_map;
        
        %fprintf('x = %i, y = %i, dx = %.1f, dy = %.1f\n', x, y, dx, dy);
      end
    end

    % Plot frame A
    if 0
    h1 = figure(1); clf
    set(h1, 'Position', [60, 60, 500, 500])
    imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
    hold on
    tmp = get(h1, 'Position');
    set(h1, 'Position', [2626, -240, tmp(3), tmp(4)])
    end

    % Plot quiver map
    %h3 = figure(3); clf
    figure(h1)
    vector_scale = 1;
    quiver(vector_map(:, :, 1), ...
           vector_map(:, :, 2), ...
           vector_map(:, :, 3), ...
           vector_map(:, :, 4), vector_scale, 'r', 'LineWidth', 2)
    drawnow
    %pause
  end
end

figure()
vector_scale = 3;
quiver(vector_map(:, :, 1), ...
       img_h - vector_map(:, :, 2), ...
       vector_map(:, :, 3), ...
       -vector_map(:, :, 4), vector_scale, 'r', 'LineWidth', 2)
xlim([1 img_w])
ylim([1 img_h])