clr;

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
  if 0
  h1 = figure(1); clf
  set(h1, 'Position', [60, 60, 500, 500])
  imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
  hold on
  tmp = get(h1, 'Position');
  set(h1, 'Position', [2626, -240, tmp(3), tmp(4)])
  end
  
  % Prepare figure 2 for showing correlation map
  h2 = figure(2); clf
  set(h2, 'Position', [780, 1117, 256, 256])
  
  % ----------------------------------------------------------------------
  %   Calculate IWs
  % ----------------------------------------------------------------------
  
  %IW_SIZE_LIST = [128 64 32];
  
  % Set interrogation window size (IW) and calculate number of IWs
  IW_SIZE = 32;
  nIW_x = floor(img_w/IW_SIZE);
  nIW_y = floor(img_h/IW_SIZE);
  
  % Allocate memory for displacement vectors
  % 3rd dim will consecutively store [x, y, dx, dy]
  vector_map = zeros(nIW_x, nIW_y, 4);
  
  %for iIW_x = 8:8
  %  for iIW_y = 18:18
  for iIW_x = 1:nIW_x
    for iIW_y = 1:nIW_y
      % Retrieve interrogation windows
      IW_x_start = (iIW_x - 1) * IW_SIZE + 1;
      IW_x_end   =  IW_x_start + IW_SIZE - 1;
      IW_y_start = (iIW_y - 1) * IW_SIZE + 1;
      IW_y_end   =  IW_y_start + IW_SIZE - 1;
      
      IW_A = A(IW_y_start:IW_y_end, IW_x_start:IW_x_end);
      IW_B = B(IW_y_start:IW_y_end, IW_x_start:IW_x_end);
             
      % Show interrogation windows on top of image
      if 0
        figure(h1)
        plot([IW_x_start IW_x_start IW_x_end IW_x_end   IW_x_start], ...
             [IW_y_start IW_y_end   IW_y_end IW_y_start IW_y_start], ...
             '-r')
      end
      
      % Perform cross-correlation
      if max(max(IW_A)) == 0 || max(max(IW_B)) == 0
        C = nan;                    % Save computation time
      else
        C = xcorr2(single(IW_B), single(IW_A));
        C = C/max(max(C))*255;      % Normalize and rescale from 0-255
      end
      
      % Show correlation map of current IW
      if 0
        figure(h2); cla
        imshow(C, jet(256), 'InitialMagnification', 300)
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
      
        % Find displacement vector (dx, dy)
        x  = IW_x_start + floor(IW_SIZE/2);
        y  = IW_y_start + floor(IW_SIZE/2);
        dx = peak_x - IW_SIZE;
        dy = peak_y - IW_SIZE;
      end
      
      % Store result in vector map
      vector_map(iIW_y, iIW_x, :) = [x, y, dx, dy];
    end
  end
  
  % Plot frame A
  if 1
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
  vector_scale = 2;
  quiver(vector_map(:, :, 1), ...
         vector_map(:, :, 2), ...
         vector_map(:, :, 3), ...
         vector_map(:, :, 4), vector_scale, 'r', 'LineWidth', 2)
  drawnow
end