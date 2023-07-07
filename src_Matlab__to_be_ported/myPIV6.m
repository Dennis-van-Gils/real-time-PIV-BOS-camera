%clr;

fDEBUG = 0;
DEBUG.x_pixel = 600;
DEBUG.y_pixel = 350;
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
        C = xcorr2(single(img_IW_B), single(img_IW_A));
        C = C/max(C(:));            % Normalize
      end
      
      % Find maximum correlation peak
      if isnan(max(C(:)))
        dx = nan; dy = nan;
      else
        [maxC, iMaxC] = max(C(:));
        [peak_y, peak_x] = ind2sub(size(C), iMaxC);

        % ----------------------------------------------------------------
        %   Perform sub-pixel algorithm, Gaussian 3 point fit
        %   But only on the smallest IW size
        % ----------------------------------------------------------------
        
        if iIW_size == nIW_SIZES
          peak_x_G = peak_x;
          peak_y_G = peak_y;
          if peak_x > 1 && peak_x < size(C, 2) - 1
            peak_x_G = peak_x - 1/2 * ...
              (log(C(peak_y, peak_x - 1)) - log(C(peak_y, peak_x + 1)) / ...
              (log(C(peak_y, peak_x - 1)) + log(C(peak_y, peak_x + 1)) - 2* ...
               log(C(peak_y, peak_x))));
             if isinf(peak_x_G); peak_x_G = peak_x; end
          end
          if peak_y > 1 && peak_y < size(C, 1) - 1
            peak_y_G = peak_y - 1/2 * ...
              (log(C(peak_y - 1, peak_x)) - log(C(peak_y + 1, peak_x)) / ...
              (log(C(peak_y - 1, peak_x)) + log(C(peak_y + 1, peak_x)) - 2* ...
               log(C(peak_y, peak_x))));
             if isinf(peak_y_G); peak_y_G = peak_y; end
          end
          peak_x = peak_x_G;
          peak_y = peak_y_G;
        end

        % Store displacement vector (x, y, dx, dy)
        dx = peak_x - IW_size + shift_x;
        dy = peak_y - IW_size + shift_y;
        
        if fDEBUG && iIW == iIW_debug && numel(C) > 1
          % Show correlation map
          figure(h3); cla
          %[Cx, Cy] = meshgrid(1:size(C, 2), 1:size(C, 1));
          %surfc(Cx, Cy, double(C*255), 'LineStyle', 'none')
          %set(gca, 'YDir', 'normal')
          imshow(C, 'InitialMagnification', 'fit', 'Colormap', jet(256))
          hold on
          plot([IW_size IW_size], [1 2*IW_size - 1], '-k', 'LineWidth', 1.5)
          plot([1 2*IW_size - 1], [IW_size IW_size], '-k', 'LineWidth', 1.5)
          plot(peak_x, peak_y, '+k', 'LineWidth', 2)
          title(['iIW = ' num2str(iIW)])
          xlabel('\delta_x [px]')
          ylabel('\delta_y [px]')
          axis on; axis tight
          
          % Show quiver
          quiver(h1_ax, IW_grid_A.x(iIW), IW_grid_A.y(iIW), ...
                 dx, dy, 2, 'Color', DEBUG.cm{iIW_size}, 'Linewidth', 2)
          drawnow
          pause
        end
      end
      
      % ------------------------------------------------------------------
      %  Store results
      % ------------------------------------------------------------------
      
      % Store result in vector map
      vector_map.dx(iIW) = dx;
      vector_map.dy(iIW) = dy;

      % Store multigrid maps
      IW_grid_As{iIW_size}  = IW_grid_A;
      IW_grid_Bs{iIW_size}  = IW_grid_B;
      vector_maps{iIW_size} = vector_map;
    end
  end
end

% ------------------------------------------------------------------------
%  Display results
% ------------------------------------------------------------------------

if 1
  % Show original full figure A
  h4 = figure(4); clf
  set(gcf, 'Position', [60, 60, 500, 500])
  imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
  hold on
  tmp = get(gcf, 'Position');
  set(gcf, 'Position', [2580, -225, tmp(3), tmp(4)])
end

vector_scale = 0;
vector_magnificier = 3;
if 1
  figure(h4)
  quiver(IW_grid_A.x, ...
         IW_grid_A.y, ...
         vector_map.dx * vector_magnificier, ...
         vector_map.dy * vector_magnificier, ...
         vector_scale, 'r', 'LineWidth', 2)
end
xlim([1 img_w])
ylim([1 img_h])

% ------------------------------------------
%  Show velocity magnitude color backdrop
% ------------------------------------------

vel_mag = sqrt(vector_map.dx.^2 + vector_map.dy.^2);

% Filter vectors based on median absolute deviation
a = mad(reshape(vel_mag, numel(vel_mag), 1));
b = nanmedian(reshape(vel_mag, numel(vel_mag), 1));
c = abs(vel_mag - b) / a;
c_mean = nanmean(nanmean(c));
c_std = nanstd(nanstd(c));
d = (vel_mag - b) / a;

outlier_threshold = 2;
iMatch = find(d > (c_mean + c_std * outlier_threshold));
%iMatch = find(vel_mag > 25);

vector_map_filtered = vector_map;
vmf_x  = IW_grid_A.x;
vmf_y  = IW_grid_A.y;
vmf_dx = vector_map.dx;
vmf_dy = vector_map.dy;
vmf_dx(iMatch) = nan;
vmf_dy(iMatch) = nan;
vmf_x(isnan(vmf_dx)) = nan;
vmf_y(isnan(vmf_dx)) = nan;
vector_map_filtered.x  = vmf_x;
vector_map_filtered.y  = vmf_y;
vector_map_filtered.dx = vmf_dx;
vector_map_filtered.dy = vmf_dy;
%vector_map_wrong.x = vector_map.dx(iMatch

vel_mag(iMatch) = nan;
vel_mag_uint8 = uint8(vel_mag / max(vel_mag(:)) * 255);
cm = jet(256);
fprintf('filtered vectors: %i\n', length(iMatch))

% Turn vel mag into rgb image
if 1
  h5 = figure(5); clf
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
  set(h4, 'Position', [2626, -240, tmp(3), tmp(4)])
  set(gca, 'Position', [0.06 0.05 .9 .95])
  axis equal
  xlim([1 img_w])
  ylim([1 img_h])
  hold on

  if 1
  quiver(vector_map_filtered.x, ...
         vector_map_filtered.y, ...
         vector_map_filtered.dx * vector_magnificier, ...
         vector_map_filtered.dy * vector_magnificier, ...
         vector_scale, 'w', 'LineWidth', 1.5)
  end
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

    quiver(vector_map_filtered.x(iVector), ...
           vector_map_filtered.y(iVector), ...
           vector_map_filtered.dx(iVector) * vector_magnificier, ...
           vector_map_filtered.dy(iVector) * vector_magnificier, ...
           vector_scale, 'LineWidth', 1.5, ...
           'Color', vel_mag_color, ...
           'MarkerFaceColor', vel_mag_color)
  end
  set(gca, 'YDir', 'reverse')
  axis tight; axis square
end