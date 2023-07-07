clr

fDEBUG = 0;
DEBUG.x_pixel = 305;
DEBUG.y_pixel = 561;
DEBUG.cm = {'r' 'g' 'm' 'w'};   % colormap of the IW borders per IW size

% DEBUG list:
% [1170, 350] using IW_SIZES = [64 32] and IW_OVERLAP = .5

% ------------------------------------------------------------------------
%   Walk over all files
% ------------------------------------------------------------------------
filelist = dir('B*.tif');
nFiles = length(filelist);

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
  clear img
  
  % Background removal
  A = A - mean(mean(A));
  B = B - mean(mean(B));
  
  if fDEBUG
    % Show original full figure A
    h1 = figure(1); clf
    set(gcf, 'Position', [2580 -225 500 500])
    imshow(imadjust(A), 'InitialMagnification', 'fit', ...
           'Colormap', bone(256));
    title('frame A')
    axis on; axis tight; hold on
    h1_ax = gca;
  
    % Show original full figure B
    h2 = figure(2); clf
    set(gcf, 'Position', get(h1, 'Position') + [520 0 0 0])
    imshow(imadjust(B), 'InitialMagnification', 'fit', ...
           'Colormap', bone(256));
    title('frame B')
    axis on; axis tight; hold on
    h2_ax = gca;
    linkaxes([h1_ax h2_ax], 'xy')

    % Prepare figure 3 for showing correlation map
    h3 = figure(3); clf
    set(h3, 'Position', [650, 685, 500, 500])
    %set(h3, 'Position', [3620, -225, 500, 500])
  end
  
  % ----------------------------------------------------------------------
  %   Initialize 
  % ----------------------------------------------------------------------
  
  % Set the IW sizes for multigrid analysis
  % Subsequent IW sizes should be the exact half of the prev IW size
  IW_SIZES   = [64 32];
  IW_OVERLAP = .5;
  
  % Allocate memory for multigrid maps
  nIW_SIZES   = length(IW_SIZES);
  IW_grid_As  = cell(nIW_SIZES, 1);
  IW_grid_Bs  = cell(nIW_SIZES, 1);
  vector_maps = cell(nIW_SIZES, 1);
  
  % ----------------------------------------------------------------------
  %   Walk over all interrogation window (IW) sizes
  % ----------------------------------------------------------------------
  
  for iIW_size = 1:nIW_SIZES
    IW_size = IW_SIZES(iIW_size);
    
    % Create IW_grid for frame A
    % Create IW_grid for frame B
    IW_grid_A = create_IW_grid(img_w, img_h, IW_size, IW_OVERLAP);
    IW_grid_B = IW_grid_A;
    
    % Already store IW_grid_A in the multigrid map for the first IW size
    if iIW_size == 1; IW_grid_As{1} = IW_grid_A; end
    
    % Allocate memory for displacement vector map
    vector_map.dx = zeros(IW_grid_A.nIWs_y, IW_grid_A.nIWs_x);
    vector_map.dy = zeros(IW_grid_A.nIWs_y, IW_grid_A.nIWs_x);
    
    % Retrieve previous displacement vector map
    if iIW_size > 1
      prev_IW_grid_A  = IW_grid_As{iIW_size - 1};
      prev_vector_map = vector_maps{iIW_size - 1};
    end
    
    % Look up the IW index to be debugged
    iIW_debug = lookup_IW_Idx(IW_grid_A, DEBUG.x_pixel, DEBUG.y_pixel);
    
    % --------------------------------------------------------------------
    %   Walk over all IWs of frame A
    % --------------------------------------------------------------------
    
    for iIW = 1:IW_grid_A.nIWs
      %fprintf('\niIW_x = %i, iIW_y = %i\n', iIW_x, iIW_y)

      % ------------------------------------------------------------------
      %   Retrieve image IW frame A
      % ------------------------------------------------------------------
      img_IW_A = A(IW_grid_A.y_range(iIW, 1):IW_grid_A.y_range(iIW, 2), ...
                   IW_grid_A.x_range(iIW, 1):IW_grid_A.x_range(iIW, 2));
      
      % ------------------------------------------------------------------
      %   Retrieve image IW frame B 
      %   Apply window shifting technique
      % ------------------------------------------------------------------
      
      if iIW_size == 1
        % First IW size, no pre-shift available
        shift_x = 0;                                    % [px]
        shift_y = 0;                                    % [px]
      else
        % Pre-shift available
        % Calculate corresponding index of prev IW
        prev_iIW = lookup_IW_Idx(prev_IW_grid_A, ...
                                 IW_grid_A.x(iIW), IW_grid_A.y(iIW));

        shift_x = prev_vector_map.dx(prev_iIW);         % [px]
        shift_y = prev_vector_map.dy(prev_iIW);         % [px]
        if isnan(shift_x); shift_x = 0; end
        if isnan(shift_y); shift_y = 0; end
      end
      
      % Calculate center of the IW in frame B using window shifting
      IW_grid_B.x_range(iIW, :) = IW_grid_B.x_range(iIW, :) + shift_x;
      IW_grid_B.y_range(iIW, :) = IW_grid_B.y_range(iIW, :) + shift_y;

      % The IW should never be shifted outside of frame B
      IW_grid_B.x_range(iIW, :) = max(IW_grid_B.x_range(iIW, :), 1);
      IW_grid_B.x_range(iIW, :) = min(IW_grid_B.x_range(iIW, :), img_w);
      IW_grid_B.y_range(iIW, :) = max(IW_grid_B.y_range(iIW, :), 1);
      IW_grid_B.y_range(iIW, :) = min(IW_grid_B.y_range(iIW, :), img_h);
      
      img_IW_B = B(IW_grid_B.y_range(iIW, 1):IW_grid_B.y_range(iIW, 2), ...
                   IW_grid_B.x_range(iIW, 1):IW_grid_B.x_range(iIW, 2));

      if fDEBUG && iIW == iIW_debug
        % Show interrogation windows on top of image
        show_IW_borders(h1_ax, IW_grid_A, iIW, DEBUG.cm{iIW_size})
        show_IW_borders(h2_ax, IW_grid_B, iIW, DEBUG.cm{iIW_size})
        
        % Zoom to the vicinity of the largest corresponding IW
        iIW_tmp = lookup_IW_Idx(IW_grid_As{1}, ...
                                DEBUG.x_pixel, DEBUG.y_pixel);
        zoom_extend = floor([-IW_size IW_size]/2);
        xlim(h1_ax, IW_grid_As{1}.x_range(iIW_tmp, :) + zoom_extend)
        ylim(h1_ax, IW_grid_As{1}.y_range(iIW_tmp, :) + zoom_extend)
        xlim(h2_ax, IW_grid_As{1}.x_range(iIW_tmp, :) + zoom_extend)
        ylim(h2_ax, IW_grid_As{1}.y_range(iIW_tmp, :) + zoom_extend)
      end

      % ------------------------------------------------------------------
      %   Perform cross-correlation
      % ------------------------------------------------------------------
      
      if max(img_IW_A(:)) == 0 || max(img_IW_B(:)) == 0
        C = nan;                    % Save computation time
      else
        C = xcorr2(double(img_IW_B), double(img_IW_A));
        C = C/max(C(:));            % Normalize
      end
      
      % Find maximum correlation peak
      if isnan(max(C(:)))
        dx = nan; dy = nan;
      else
        [maxC, iMaxC] = max(C(:));
        [peak_y, peak_x] = ind2sub(size(C), iMaxC);

        % ----------------------------------------------------------------
        %   Perform sub-pixel algorithm, 3-point Gaussian fit
        %   But only on the smallest IW size
        % ----------------------------------------------------------------
        
        if iIW_size == nIW_SIZES
          peak_x_sub = subpx_3pgf(C(peak_y, :), peak_x);
          peak_y_sub = subpx_3pgf(C(:, peak_x), peak_y);
          peak_x = peak_x_sub;
          peak_y = peak_y_sub;
        end

        % Calculate displacement vector
        dx = peak_x - IW_size + shift_x;
        dy = peak_y - IW_size + shift_y;
        
        if fDEBUG && iIW == iIW_debug && numel(C) > 1
          % Show correlation map
          figure(h3); cla
          imshow(C, 'InitialMagnification', 'fit', 'Colormap', jet(256))
          hold on
          plot([IW_size IW_size], [1 2*IW_size-1], '-k', 'LineWidth', 1.5)
          plot([1 2*IW_size-1], [IW_size IW_size], '-k', 'LineWidth', 1.5)
          plot(peak_x, peak_y, '+k', 'LineWidth', 2)
          title(['iIW = ' num2str(iIW)])
          xlabel('\delta_x [px]')
          ylabel('\delta_y [px]')
          axis on; axis tight
          
          % Show quiver
          quiver(h1_ax, IW_grid_A.x(iIW), IW_grid_A.y(iIW), ...
                 dx, dy, 2, 'Color', DEBUG.cm{iIW_size}, 'Linewidth', 2)
          drawnow
        end
      end
      
      % ------------------------------------------------------------------
      %  Store IW results
      % ------------------------------------------------------------------
      
      % Store result in vector map
      vector_map.dx(iIW) = dx;
      vector_map.dy(iIW) = dy;
      
      % ------------------------------------------------------------------
      % ------------------------------------------------------------------
      %  End: all IWs of the current IW_grid are processed
      % ------------------------------------------------------------------
      % ------------------------------------------------------------------
    end
    
    % --------------------------------------------------------------------
    %   Calculate derived quantities of the full maps
    % --------------------------------------------------------------------
    
    % Calculate velocity magnitude
    vector_map.vel_mag = sqrt(vector_map.dx.^2 + vector_map.dy.^2);

    % Plot velocity magnitude
    h = figure;
    imshow(vector_map.vel_mag, [0 max(vector_map.vel_mag(:))], ...
           'InitialMagnification', 'fit', 'Colormap', jet(256))
    title(['IW size = ' num2str(IW_size) ...
           ', overlap = ' num2str(IW_OVERLAP)])
    axis on; axis tight
    
    % --------------------------------------------------------------------
    %  Detect bad vectors
    % --------------------------------------------------------------------
    
    % Filter vectors based on median absolute deviation
    outlier_threshold = 2;
    c = (vector_map.vel_mag(:) - nanmedian(vector_map.vel_mag(:))) / ...
        mad(vector_map.vel_mag(:));
    iRemove = find(c > (nanmean(abs(c)) + ...
                        nanstd(abs(c)) * outlier_threshold));
                      
    [i, j] = ind2sub([IW_grid_A.nIWs_y, IW_grid_A.nIWs_x], iRemove);
    hold on
    plot(j, i, 'xk', 'MarkerSize', 6, 'LineWidth', 2)

    vector_map_filt = vector_map;
    vmf_x  = IW_grid_A.x;
    vmf_y  = IW_grid_A.y;
    vmf_dx = vector_map.dx;
    vmf_dy = vector_map.dy;
    vmf_dx(iRemove) = nan;
    vmf_dy(iRemove) = nan;
    vmf_x(isnan(vmf_dx)) = nan;
    vmf_y(isnan(vmf_dx)) = nan;
    vector_map_filt.x  = vmf_x;
    vector_map_filt.y  = vmf_y;
    vector_map_filt.dx = vmf_dx;
    vector_map_filt.dy = vmf_dy;
    vector_map_filt.vel_mag(iRemove) = nan;
    %vector_map_wrong.x = vector_map.dx(iRemove

    fprintf('filtered vectors: %i\n', length(iRemove))
    
    %disp('')
    %pause

    
    % --------------------------------------------------------------------
    %  Store multigrid maps
    % --------------------------------------------------------------------
    
    IW_grid_As{iIW_size}  = IW_grid_A;
    IW_grid_Bs{iIW_size}  = IW_grid_B;
    vector_maps{iIW_size} = vector_map;
  end
end

% DEBUG: working out multipeak detection
% if fDEBUG && iIW == iIW_debug && numel(C) > 1
%     figure(5); clf
%     C_BW = imextendedmax(C, .2, 8);
%     imagesc(C_BW)
%     axis square
%     local_peaks = regionprops(C_BW, 'PixelIdxList');
%     local_peak_intensity = zeros(length(local_peaks), 1);
%     for iPeak = 1:length(local_peaks)
%       local_peak_intensity(iPeak) = ...
%         max(C(local_peaks(iPeak).PixelIdxList));
%     end
%     disp('stop here')
% end

% ------------------------------------------------------------------------
%  Display results
% ------------------------------------------------------------------------
vector_scale = 0;
vector_magnificier = 3;

if 1
  % Show original full figure A
  h6 = figure(6); clf
  set(gcf, 'Position', [60, 60, 500, 500])
  imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
  hold on
  tmp = get(gcf, 'Position');
  set(gcf, 'Position', [2580, -225, tmp(3), tmp(4)])

  % Show simple quiver map
  quiver(IW_grid_A.x, ...
         IW_grid_A.y, ...
         vector_map.dx * vector_magnificier, ...
         vector_map.dy * vector_magnificier, ...
         vector_scale, 'r', 'LineWidth', 2)
  xlim([1 img_w])
  ylim([1 img_h])
end

% ------------------------------------------
%  Show velocity magnitude color backdrop
% ------------------------------------------

vel_mag       = vector_map_filt.vel_mag;
vel_mag_uint8 = uint8(vel_mag / max(vel_mag(:)) * 255);
cm = jet(256);

% Turn vel mag into rgb image
if 1
  h7 = figure(7); clf
  [px_x, px_y] = meshgrid(1:img_w, 1:img_h);
  %vel_mag(isnan(vel_mag)) = 0;
  vel_mag_img = interp2(IW_grid_A.x, ...
                        IW_grid_A.y, ...
                        vel_mag, ...
                        px_x, px_y, ...
                        'nearest');
  vel_mag_img = ind2rgb(uint8(vel_mag_img/max(vel_mag_img(:))*255), jet(256));
  img=imfuse(vel_mag_img, ind2rgb(imadjust(A), gray(256)), 'blend');
  imagesc(img)
  set(gcf, 'Position', [2626, -240, tmp(3), tmp(4)])
  set(gca, 'Position', [0.06 0.05 .9 .95])
  axis equal
  xlim([1 img_w])
  ylim([1 img_h])
  hold on

  quiver(vector_map_filt.x, ...
         vector_map_filt.y, ...
         vector_map_filt.dx * vector_magnificier, ...
         vector_map_filt.dy * vector_magnificier, ...
         vector_scale, 'w', 'LineWidth', 1.5)
end

% ------------------------------------------
%  Individually colored vectors
% ------------------------------------------

if 0
  % Show original full figure A
  h8 = figure(8); clf
  set(gcf, 'Position', [60, 60, 500, 500])
  imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
  hold on
  tmp = get(gcf, 'Position');
  set(gcf, 'Position', [2580, -225, tmp(3), tmp(4)])
  
  for iVector = 1:numel(vector_map.dx)
    vel_mag_color = cm(vel_mag_uint8(iVector) + 1, :);

    quiver(vector_map_filt.x(iVector), ...
           vector_map_filt.y(iVector), ...
           vector_map_filt.dx(iVector) * vector_magnificier, ...
           vector_map_filt.dy(iVector) * vector_magnificier, ...
           vector_scale, 'LineWidth', 1.5, ...
           'Color', vel_mag_color, ...
           'MarkerFaceColor', vel_mag_color)
  end
  set(gca, 'YDir', 'reverse')
  axis tight; axis square
end