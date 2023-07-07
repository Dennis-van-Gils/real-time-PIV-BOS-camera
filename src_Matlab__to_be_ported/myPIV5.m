%clr;

fSHOW_IWs = 1;
fSHOW_XCOR = 0;

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
  clear img
  
  % Background removal
  A = A - mean(mean(A));
  B = B - mean(mean(B));
  
  if fSHOW_IWs
    % Show original full figure A
    h1 = figure(1); clf
    set(gcf, 'Position', [60, 60, 500, 500])
    imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
    tmp = get(gcf, 'Position');
    set(gcf, 'Position', [2580, -225, tmp(3), tmp(4)])
    h1_ax = gca;
    hold on
  
    % Show original full figure B
    h2 = figure(2); clf
    imshow(imadjust(B), 'InitialMagnification', 80, 'Colormap', bone(256));
    set(gcf, 'Position', get(h1, 'Position'))
    h2_ax = gca;
    hold on
  end
  
  if fSHOW_XCOR
    % Prepare figure 3 for showing correlation map
    h3 = figure(3); clf
    set(h3, 'Position', [650, 685, 640, 640])
  end
  
  % ----------------------------------------------------------------------
  %   Calculate IWs
  % ----------------------------------------------------------------------
  
  % Set the IW sizes for multigrid analysis
  % Subsequent IW sizes should be the exact half of the prev IW size
  IW_SIZES   = [64 32 16];
  IW_OVERLAP = .5;
  nIW_SIZES = length(IW_SIZES);
  
  % Allocate memory for multigrid maps
  IW_grid_As  = cell(nIW_SIZES, 1);
  vector_maps = cell(nIW_SIZES, 1);
  
  for iIW_SIZE = 1:nIW_SIZES
    fLastIW_SIZE = (iIW_SIZE == nIW_SIZES);   % Are we at the last IW_SIZE?
  
    % Set interrogation window size (IW)
    IW_size = IW_SIZES(iIW_SIZE);
    
    % Create IW_grid for the source frame A
    IW_grid_A = create_IW_grid(img_w, img_h, IW_size, IW_OVERLAP);
  
    % Allocate memory for displacement vector map
    vector_map.dx = zeros(IW_grid_A.nIWs_y, IW_grid_A.nIWs_x);
    vector_map.dy = zeros(IW_grid_A.nIWs_y, IW_grid_A.nIWs_x);
    
    % Retrieve previous displacement vector map
    if iIW_SIZE > 1
      prev_IW_grid_A  = IW_grid_As{iIW_SIZE - 1};
      prev_vector_map = vector_maps{iIW_SIZE - 1};
    end
    
    % Walk over all IWs
    for iIW = 1:IW_grid_A.nIWs
      %fprintf('\niIW_x = %i, iIW_y = %i\n', iIW_x, iIW_y)

      % ----------------------------------------------------------------
      %   Retrieve image interrogation window frame A
      % ----------------------------------------------------------------
      A_x1 = IW_grid_A.x_range(iIW, 1);
      A_x2 = IW_grid_A.x_range(iIW, 2);
      A_y1 = IW_grid_A.y_range(iIW, 1);
      A_y2 = IW_grid_A.y_range(iIW, 2);
      img_IW_A = A(A_y1:A_y2, A_x1:A_x2);
      
      % ----------------------------------------------------------------
      %   Retrieve image interrogation window frame B
      %   Apply window shifting technique
      % ----------------------------------------------------------------                

      if iIW_SIZE == 1
        % First IW size, no pre-shift available
        shift_x = 0;
        shift_y = 0;
      else
        % Pre-shift available

        % Calculate corresponding index of prev IW
        prev_iIW = lookup_IW_Idx(prev_IW_grid_A, ...
                                 IW_grid_A.x(iIW), ...
                                 IW_grid_A.y(iIW));

        shift_x = prev_vector_map.dx(prev_iIW);
        shift_y = prev_vector_map.dy(prev_iIW);

        if isnan(shift_x); shift_x = 0; end
        if isnan(shift_y); shift_y = 0; end
      end

      IW_B.x_range = IW_grid_A.x_range(iIW, :) + shift_x;
      IW_B.y_range = IW_grid_A.y_range(iIW, :) + shift_y;

      % The IW should never be shifted outside of frame B
      IW_B.x_range = max(IW_B.x_range, 1);
      IW_B.x_range = min(IW_B.x_range, img_w);
      IW_B.y_range = max(IW_B.y_range, 1);
      IW_B.y_range = min(IW_B.y_range, img_h);
      
      B_x1 = IW_B.x_range(1);
      B_x2 = IW_B.x_range(2);
      B_y1 = IW_B.y_range(1);
      B_y2 = IW_B.y_range(2);
      img_IW_B = B(B_y1:B_y2, B_x1:B_x2);
                 
      % Show interrogation windows on top of image
      if fSHOW_IWs && iIW == lookup_IW_Idx(IW_grid_A, 390, 507)
        switch iIW_SIZE
          case 1; myColor = 'r';
          case 2; myColor = 'y';
          otherwise; myColor = 'm';
        end
        plot(h1_ax, [A_x1 A_x1 A_x2 A_x2 A_x1] + ...
                    [-.5 -.5 .5 .5 -.5], ... % esthetiques
                    [A_y1 A_y2 A_y2 A_y1 A_y1] + ...
                    [-.5 .5 .5 -.5 -.5], ... % esthetiques
                    '-', 'Color', myColor, 'LineWidth', 2)
        plot(h2_ax, [B_x1 B_x1 B_x2 B_x2 B_x1] + ...,
                    [-.5 -.5 .5 .5 -.5], ... % esthetiques
                    [B_y1 B_y2 B_y2 B_y1 B_y1] + ...
                    [-.5 .5 .5 -.5 -.5], ... % esthetiques
                    '-', 'Color', myColor, 'LineWidth', 2)
      end

      % Perform cross-correlation
      if max(max(img_IW_A)) == 0 || max(max(img_IW_B)) == 0
        C = nan;                    % Save computation time
      else
        C = xcorr2(single(img_IW_B), single(img_IW_A));
        C = C/max(max(C));          % Normalize
      end
      
      % Find maximum correlation peak
      if isnan(max(max(C)))        
        %x = nan; y = nan;
        dx = nan; dy = nan;
      else
        [maxC, iMaxC] = max(C(:));
        [peak_y, peak_x] = ind2sub(size(C), iMaxC);

        % Perform sub-pixel algorithm, Gaussian 3 point fit
        % But only on the smallest IW size
        if fLastIW_SIZE
          peak_x_G = peak_x;
          peak_y_G = peak_y;
          if peak_x > 1 && peak_x < IW_size * 2 - 2
            peak_x_G = peak_x - 1/2 * ...
              (log(C(peak_y, peak_x - 1)) - log(C(peak_y, peak_x + 1)) / ...
              (log(C(peak_y, peak_x - 1)) + log(C(peak_y, peak_x + 1)) - 2* ...
               log(C(peak_y, peak_x))));
             if isinf(peak_x_G); peak_x_G = peak_x; end
          end
          if peak_y > 1 && peak_y < IW_size * 2 - 2
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
        
        if fSHOW_XCOR && numel(C) > 1
          % Show correlation map of current IW
          figure(h3); cla
          imshow(C*255, jet(256), 'InitialMagnification', 300)
          %[Cx, Cy] = meshgrid(1:size(C, 2), 1:size(C, 1));
          %surfc(Cx, Cy, double(C/max(max(C))*255), 'LineStyle', 'none')
          %set(gca, 'YDir', 'normal')
          hold on
          plot([IW_size IW_size], [1 2*IW_size - 1], '-k', 'LineWidth', 1.5)
          plot([1 2*IW_size - 1], [IW_size IW_size], '-k', 'LineWidth', 1.5)
          plot(peak_x, peak_y, '+k', 'LineWidth', 2)
          title(['iIW = ' num2str(iIW)])
          xlabel('\delta_x [px]')
          ylabel('\delta_y [px]')
          axis tight; axis on
          drawnow
        end
        
        if fSHOW_IWs && iIW == lookup_IW_Idx(IW_grid_A, 390, 507)
          % Show quiver
          quiver(h1_ax, IW_grid_A.x(iIW), IW_grid_A.y(iIW), ...
                 dx, dy, 2, 'Color', myColor, 'Linewidth', 2)
        end
      end
      
      if iIW == 18
        disp('')
      end

      % Store result in vector map
      vector_map.dx(iIW) = dx;
      vector_map.dy(iIW) = dy;

      % Store vector map in vector maps
      IW_grid_As{iIW_SIZE} = IW_grid_A;
      vector_maps{iIW_SIZE} = vector_map;

      %fprintf('x = %i, y = %i, dx = %.1f, dy = %.1f\n', x, y, dx, dy);
    end
  end
end

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

return

vel_mag = sqrt(vector_map.dx.^2 + vector_map.dy.^2);

% Filter vectors based on median absolute deviation
a = mad(reshape(vel_mag, numel(vel_mag), 1));
b = nanmedian(reshape(vel_mag, numel(vel_mag), 1));
c = abs(vel_mag - b) / a;
c_mean = nanmean(nanmean(c));
c_std = nanstd(nanstd(c));
d = (vel_mag - b) / a;

outlier_threshold = .5;
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

vel_mag(iMatch) = nan;
vel_mag_uint8 = uint8(vel_mag / max(max(vel_mag)) * 255);
cm = jet(255);

% Turn vel mag into rgb image
if 0
  h4 = figure(4); clf
  [px_x, px_y] = meshgrid(1:img_w, 1:img_h);
  %vel_mag(isnan(vel_mag)) = 0;
  vel_mag_img = interp2(IW_grid_A.x', ...
                        IW_grid_A.y', ...
                        vel_mag', ...
                        px_x, px_y, ...
                        'nearest');
  vel_mag_img = ind2rgb(uint8(vel_mag_img/max(max(vel_mag_img))*255), jet(255));
  img=imfuse(vel_mag_img, ind2rgb(imadjust(A), gray(255)), 'blend');
  imagesc(img)
  set(h4, 'Position', [2626, -240, tmp(3), tmp(4)])
  set(gca, 'Position', [0.06 0.05 .9 .95])
  axis equal
  xlim([1 img_w])
  ylim([1 img_h])
  hold on
end

if 0
  figure(3); clf
  h_pc = pcolor(vector_map(:, :, 1), vector_map(:, :, 2), ...
                double(vel_mag_uint8));
  set(h_pc, 'FaceAlpha', .5)
  set(gca, 'YDir', 'reverse')
  set(gca, 'Color', 'none')
  colormap(jet(255))
  shading interp
  axis equal
  xlim([1 img_w])
  ylim([1 img_h])
  hold on
end

if 0
  quiver(vector_map_filtered.x, ...
         vector_map_filtered.y, ...
         vector_map_filtered.dx' * vector_magnificier, ...
         vector_map_filtered.dy' * vector_magnificier, ...
         vector_scale, 'w', 'LineWidth', 1.5)
end

if 0
  quiverwcolorbar(vector_map_filtered(:, :, 1), ...
                  vector_map_filtered(:, :, 2), ...
                  vector_map_filtered(:, :, 3) * vector_magnificier, ...
                  vector_map_filtered(:, :, 4) * vector_magnificier, ...
                  vector_scale, 'bounds', [0 max(max(vel_mag))])
end

if 0
  for iCol = 1:size(vector_map, 1)
    for iRow = 1:size(vector_map, 2)
      this_vel_mag_uint8 = vel_mag_uint8(iCol, iRow);
      this_vel_mag_color = cm(this_vel_mag_uint8 + 1, :);

      
%       quiverwcolorbar(vector_map_filtered(iCol, iRow, 1), ...
%                       vector_map_filtered(iCol, iRow, 2), ...
%                       vector_map_filtered(iCol, iRow, 3) * vector_magnificier, ...
%                       vector_map_filtered(iCol, iRow, 4) * vector_magnificier, ...
%                       vector_scale, 'bounds', [1 255])
      quiver(vector_map_filtered(iCol, iRow, 1), ...
             vector_map_filtered(iCol, iRow, 2), ...
             vector_map_filtered(iCol, iRow, 3) * vector_magnificier, ...
             vector_map_filtered(iCol, iRow, 4) * vector_magnificier, ...
             vector_scale, 'r', 'LineWidth', 2, ...
             'Color', this_vel_mag_color, 'MarkerFaceColor', this_vel_mag_color)
    end
  end

end