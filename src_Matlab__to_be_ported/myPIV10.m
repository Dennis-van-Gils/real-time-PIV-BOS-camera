clr

fPROFILER = 0;
if fPROFILER
  profile clear                                                             %#ok<*UNRCH>
  profile on
end

fDEBUG = 0;
DEBUG.x_pixel = 558;
DEBUG.y_pixel = 400;
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
  img_blend = imread(fn);
  
  % Read double image and split into frames A and B
  [img_h, img_w] = size(img_blend);
  A = img_blend(1:img_h/2, :);
  B = img_blend(img_h/2+1:end, :);
  img_h = img_h/2;
  clear img_blend
  
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
  %IW_SIZES   = [128 96 64 48 32];
  IW_SIZES   = [64 32];
  IW_OVERLAP = .5;
  
  % Allocate memory for multigrid maps
  nIW_SIZES   = length(IW_SIZES);
  IW_grid_As  = cell(nIW_SIZES, 1);
  IW_grid_Bs  = cell(nIW_SIZES, 1);
  VMs         = cell(nIW_SIZES, 1);
  
  %% ---------------------------------------------------------------------
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
    VM.descr = 'unfiltered';
    VM.x  = IW_grid_A.x;
    VM.y  = IW_grid_A.y;
    VM.dx = zeros(IW_grid_A.nIWs_y, IW_grid_A.nIWs_x);
    VM.dy = zeros(IW_grid_A.nIWs_y, IW_grid_A.nIWs_x);
    
    % Look up the IW index to be debugged
    iIW_debug = lookup_IW_Idx(IW_grid_A, DEBUG.x_pixel, DEBUG.y_pixel);
    
    %% -------------------------------------------------------------------
    %   Walk over all IWs
    % --------------------------------------------------------------------
    
    for iIW = 1:IW_grid_A.nIWs
      %fprintf('\niIW = %i\n', iIW)
      
      %% -----------------------------------------------------------------
      %   Calculate IW of frame B 
      %   Apply window shifting technique
      % ------------------------------------------------------------------
      
      if iIW_size == 1
        % First IW size, no pre-shift available
        shift_x = 0;                                          % [px]
        shift_y = 0;                                          % [px]
      else
        % Pre-shift available
        % Calculate corresponding index of the IW in the larger parent grid
        iIW_parent = lookup_IW_Idx(IW_grid_As{iIW_size - 1}, ...
                                   IW_grid_A.x(iIW), IW_grid_A.y(iIW));
        % Retrieve the pre-shift
        shift_x = round(VMs{iIW_size - 1}.dx(iIW_parent));    % [px]
        shift_y = round(VMs{iIW_size - 1}.dy(iIW_parent));    % [px]
        if isnan(shift_x); shift_x = 0; end
        if isnan(shift_y); shift_y = 0; end
      
        % Calculate new center and range of the shifted IW in frame B
        IW_grid_B.x(iIW)          = IW_grid_B.x(iIW)          + shift_x;
        IW_grid_B.y(iIW)          = IW_grid_B.y(iIW)          + shift_y;
        IW_grid_B.x_range(iIW, :) = IW_grid_B.x_range(iIW, :) + shift_x;
        IW_grid_B.y_range(iIW, :) = IW_grid_B.y_range(iIW, :) + shift_y;

        % The IW should never be shifted outside of frame B.
        % When it does, equally resize the IWs of both frames A and B such
        % that the resized IW of frame B still fits in frame B
        if IW_grid_B.x_range(iIW, 1) < 1
          IW_grid_B.x_range(iIW, 1) = 1;
          IW_grid_A.x_range(iIW, 1) = 1 - shift_x;
        end
        if IW_grid_B.y_range(iIW, 1) < 1
          IW_grid_B.y_range(iIW, 1) = 1;
          IW_grid_A.y_range(iIW, 1) = 1 - shift_y;
        end
        if IW_grid_B.x_range(iIW, 2) > img_w
          IW_grid_B.x_range(iIW, 2) = img_w;
          IW_grid_A.x_range(iIW, 2) = img_w - shift_x;
        end
        if IW_grid_B.y_range(iIW, 2) > img_h
          IW_grid_B.y_range(iIW, 2) = img_h;
          IW_grid_A.y_range(iIW, 2) = img_h - shift_y;
        end
      end
      
      if fDEBUG && iIW == iIW_debug
        % Show interrogation windows on top of image
        show_IW_borders(h1_ax, IW_grid_A, iIW, DEBUG.cm{iIW_size})
        show_IW_borders(h2_ax, IW_grid_B, iIW, DEBUG.cm{iIW_size})
        
        plot(h2_ax, IW_grid_B.x(iIW), IW_grid_B.y(iIW), 'x', ...
             'Color', DEBUG.cm{iIW_size}, 'MarkerSize', 8)
        
        % Zoom to the vicinity of the largest corresponding IW
        iIW_tmp = lookup_IW_Idx(IW_grid_As{1}, ...
                                DEBUG.x_pixel, DEBUG.y_pixel);
        zoom_extend = floor([-IW_size IW_size]/2);
        xlim(h1_ax, IW_grid_As{1}.x_range(iIW_tmp, :) + zoom_extend)
        ylim(h1_ax, IW_grid_As{1}.y_range(iIW_tmp, :) + zoom_extend)
        xlim(h2_ax, IW_grid_As{1}.x_range(iIW_tmp, :) + zoom_extend)
        ylim(h2_ax, IW_grid_As{1}.y_range(iIW_tmp, :) + zoom_extend)
      end
      
      %% -----------------------------------------------------------------
      %   Retrieve images of IW frame A and IW frame B
      % ------------------------------------------------------------------
      
      img_IW_A = A(IW_grid_A.y_range(iIW, 1):IW_grid_A.y_range(iIW, 2), ...
                   IW_grid_A.x_range(iIW, 1):IW_grid_A.x_range(iIW, 2));
      img_IW_B = B(IW_grid_B.y_range(iIW, 1):IW_grid_B.y_range(iIW, 2), ...
                   IW_grid_B.x_range(iIW, 1):IW_grid_B.x_range(iIW, 2));

      %% -----------------------------------------------------------------
      %   Perform cross-correlation
      % ------------------------------------------------------------------

      if isempty(img_IW_A(:)) || ...
          max(img_IW_A(:)) == 0 || max(img_IW_B(:)) == 0
        C = nan;                        % Save computation time
      else
        %C = xcorr2(double(img_IW_B), ...
        %           double(img_IW_A));   % Slow but accurate
        C = xcorr2(single(img_IW_B), ...
                   single(img_IW_A));   % Fast but slightly less accurate
        C = C/max(C(:));                % Normalize
      end
      
      % Find maximum correlation peak
      if isnan(max(C(:)))
        dx = nan; dy = nan;
      else
        [maxC, iMaxC] = max(C(:));
        [peak_y, peak_x] = ind2sub(size(C), iMaxC);

        % Sub-pixel resolution algorithm, 3-point Gaussian fit
        [peak_x, peak_y] = subpx_3pgf_2D(C, peak_x, peak_y);
        
        % Calculate displacement vector
        dx = peak_x - floor(size(C, 2)/2 + 1) + shift_x;
        dy = peak_y - floor(size(C, 1)/2 + 1) + shift_y;
      end
      
      % Store result in vector map
      VM.dx(iIW) = dx;
      VM.dy(iIW) = dy;
      
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
        quiver(h1_ax, VM.x(iIW), VM.y(iIW), VM.dx(iIW), VM.dy(iIW), ...
               2, 'Color', DEBUG.cm{iIW_size}, 'Linewidth', 2)
        drawnow
      end
    end
    %%
    
    
    % ********************************************************************
    %
    %         All IWs of the current IW_grid have been processed
    %
    % ********************************************************************
    
    
    
    %% -------------------------------------------------------------------
    %   Calculate derived quantities of the full maps
    % --------------------------------------------------------------------

    VM.magn  = sqrt(VM.dx.^2 + VM.dy.^2); % velocity magnitude  [px/frame]
    VM.angle = atan2(VM.dy, VM.dx);       % vector angle        [rad]
    VM.angle(VM.magn == 0) = nan;         % set undefined angle to nan

    % Plot velocity magnitude
    h = figure;
    imshow(VM.magn, [0 max(VM.magn(:))], ...
           'InitialMagnification', 'fit', 'Colormap', jet(256))
    title(['IW size = ' num2str(IW_size) ...
           ', overlap = ' num2str(IW_OVERLAP)])
    axis on; axis tight; hold on
    
    %% --------------------------------------------------------------------
    %   DEBUG: quicksave/quickload results
    % --------------------------------------------------------------------
    
    %save('full_mem_dump.mat')
    %return
    
    %% -------------------------------------------------------------------
    %   Process bad vectors
    % --------------------------------------------------------------------
    
    VM_filt = VM;
    VM_filt.iReplaced = [];
    VM_filt.nReplaced = 0;
    VM_filt.descr = 'unfiltered';
    VM_filt.iReplacedCum = [];
    VM_filt.nReplacedCum = 0;
    
    % Filter vectors based on median absolute deviation of the magnitude
    if 1
      outlier_threshold = 2.5;  % 2
      c = (VM.magn(:) - nanmedian(VM.magn(:))) / mad(VM.magn(:));
      iReplaced = find(c > (nanmean(abs(c)) + ...
                            nanstd(abs(c)) * outlier_threshold));
      nReplaced = length(iReplaced);
      clear c

      % Plot bad vector indicators
      [i, j] = ind2sub([IW_grid_A.nIWs_y, IW_grid_A.nIWs_x], iReplaced);
      plot(j, i, 'xw', 'MarkerSize', 6, 'LineWidth', 2)

      VM_filt.descr = 'filtered';
      VM_filt.dx(iReplaced)   = nan;
      VM_filt.dy(iReplaced)   = nan;
      VM_filt.magn(iReplaced) = nan;
      VM_filt.iReplaced       = iReplaced;
      VM_filt.nReplaced       = nReplaced;
      VM_filt.iReplacedCum    = [VM_filt.iReplacedCum; iReplaced];
      VM_filt.nReplacedCum    = VM_filt.nReplacedCum + nReplaced;
      
      fprintf('filtered vectors #1: %i\n', nReplaced)
    end
    
    % Replace bad vectors by nearest neighborhood
    if 1
      VM_filt = replace_bad_vectors_by_nn(VM_filt);
    end
    
    % Filter vectors based on LP filter on angle and magnitude
    if 1
      if IW_size >= 128
        [iBadAngle, iBadMagn] = detect_bad_vectors(VM_filt, 180, 10);
      elseif IW_size >= 64
        [iBadAngle, iBadMagn] = detect_bad_vectors(VM_filt, 40, 4);
      elseif IW_size >= 32
        [iBadAngle, iBadMagn] = detect_bad_vectors(VM_filt, 20, 4);
      end
      
      iReplaced = unique([iBadAngle; iBadMagn]);
      nReplaced = length(iReplaced);
      
      % Plot bad vector indicators
      [i, j] = ind2sub([IW_grid_A.nIWs_y, IW_grid_A.nIWs_x], iBadAngle);
      plot(j, i, 'xr', 'MarkerSize', 6, 'LineWidth', 2)
      [i, j] = ind2sub([IW_grid_A.nIWs_y, IW_grid_A.nIWs_x], iBadMagn);
      plot(j, i, 'xk', 'MarkerSize', 6, 'LineWidth', 2)

      VM_filt.descr = 'filtered';
      VM_filt.dx(iReplaced)   = nan;
      VM_filt.dy(iReplaced)   = nan;
      VM_filt.magn(iReplaced) = nan;
      VM_filt.iReplaced = iReplaced;
      VM_filt.nReplaced = nReplaced;
      VM_filt.iReplacedCum    = [VM_filt.iReplacedCum; iReplaced];
      VM_filt.nReplacedCum    = VM_filt.nReplacedCum + nReplaced;
      
      fprintf('filtered vectors #2: %i\n', nReplaced)
    end

    % Replace bad vectors by nearest neighborhood
    if 1
      VM_filt = replace_bad_vectors_by_nn(VM_filt);
    end
    
    %% -------------------------------------------------------------------
    %   Revisit bad vectors and try the 2, 3 or 4th max correlation peak
    % --------------------------------------------------------------------
    
    if 0
    for iNr = 1:nRemove
      iIW = iRemove(iNr);
      
      img_IW_A = A(IW_grid_A.y_range(iIW, 1):IW_grid_A.y_range(iIW, 2), ...
                   IW_grid_A.x_range(iIW, 1):IW_grid_A.x_range(iIW, 2));
      img_IW_B = B(IW_grid_B.y_range(iIW, 1):IW_grid_B.y_range(iIW, 2), ...
                   IW_grid_B.x_range(iIW, 1):IW_grid_B.x_range(iIW, 2));
                 
      C = xcorr2(single(img_IW_B), single(img_IW_A));
      C = C/max(C(:));            % Normalize

      [maxC, iMaxC] = max(C(:));
      [peak_y, peak_x] = ind2sub(size(C), iMaxC);
      
      % Sub-pixel resolution algorithm, 3-point Gaussian fit
      [peak_x, peak_y] = subpx_3pgf_2D(C, peak_x, peak_y);
        
      % Calculate displacement vector
      dx = peak_x - floor(size(C, 2)/2 + 1) + shift_x;
      dy = peak_y - floor(size(C, 1)/2 + 1) + shift_y;
      
      % DEBUG: working out multipeak detection
      figure(5); clf
      C_BW = imextendedmax(C, .2, 8);
      imagesc(C_BW)
      axis square
      h5_ax = gca;
      local_peaks = regionprops(C_BW, 'PixelIdxList');
      local_peak_intensity = zeros(length(local_peaks), 1);
      for iPeak = 1:length(local_peaks)
        local_peak_intensity(iPeak) = ...
          max(C(local_peaks(iPeak).PixelIdxList));
      end
      
      % Show correlation map
      figure(3); cla
      imshow(double(C), 'InitialMagnification', 'fit', 'Colormap', jet(256))
      hold on
      plot([IW_size IW_size], [1 2*IW_size-1], '-k', 'LineWidth', 1.5)
      plot([1 2*IW_size-1], [IW_size IW_size], '-k', 'LineWidth', 1.5)
      plot(peak_x, peak_y, '+k', 'LineWidth', 2)
      h3_ax = gca;
      title(['iIW = ' num2str(iIW)])
      xlabel('\delta_x [px]')
      ylabel('\delta_y [px]')
      axis on; axis tight

      linkaxes([h3_ax h5_ax], 'xy')
      
      drawnow
      pause
    end
    end % if 0
    
    
    %% -------------------------------------------------------------------
    %   Store multigrid maps
    % --------------------------------------------------------------------
    
    IW_grid_As{iIW_size} = IW_grid_A;
    IW_grid_Bs{iIW_size} = IW_grid_B;
    
    % Store the filtered vector map if we are not yet at the final IW size
    if iIW_size == nIW_SIZES
      VMs{iIW_size} = VM;
    else
      VMs{iIW_size} = VM_filt;
    end
  end
end
%%
    
    
% ************************************************************************
% 
%                             Display results
% 
% ************************************************************************



%% -----------------------------------------------------------------------
%   Show original image A with unfiltered vector map on top
% ------------------------------------------------------------------------
quiverX = 3;
quiverScale = 0;

if 1
  h6 = figure(6); clf
  set(gcf, 'Position', [2580, -225, 1280, 900])
  imshow(imadjust(A), 'InitialMagnification', 'fit', ...
         'Colormap', bone(256));
  set(gca, 'Position', [0.06 0.05 .9 .91])
  axis on; axis tight; hold on
  
  quiver(VM.x, VM.y, VM.dx * quiverX, VM.dy * quiverX, ...
         quiverScale, 'r', 'LineWidth', 2)
  xlim([1 img_w]); ylim([1 img_h])
  title(VM.descr)
  %xlim([248 618]); ylim([262 538])
end

%% -----------------------------------------------------------------------
%   Show velocity magnitude color backdrop
% ------------------------------------------------------------------------

if 1
  %thisVM = VM;
  thisVM = VM_filt;
  
  % Do not plot zero magnitude vectors
  thisVM.x(isnan(thisVM.dx)) = nan;
  thisVM.y(isnan(thisVM.dy)) = nan;

  % Turn velocity magnitude map into rgb image  
  [px_x, px_y] = meshgrid(1:img_w, 1:img_h);
 
  magn_nan2zero = thisVM.magn;
  magn_nan2zero(isnan(magn_nan2zero)) = 0;
  
  % Input grid [.x, .y] must not contain NaNs! Hence, use original IW_grid.
  img_magn = interp2(IW_grid_A.x, IW_grid_A.y, magn_nan2zero, px_x, px_y, ...
                     'nearest');
  img_magn  = ind2rgb(uint8(img_magn/max(img_magn(:))*255), jet(256));
  img_blend = imfuse(img_magn, ind2rgb(imadjust(A), gray(256)), 'blend');
  
  h7 = figure(7); clf
  set(gcf, 'Position', [2580, -225, 1280, 900])
  imshow(img_blend, 'InitialMagnification', 'fit')
  set(gca, 'Position', [0.06 0.05 .9 .91])
  axis on; axis tight; hold on
  
  quiver(thisVM.x, thisVM.y, thisVM.dx * quiverX, thisVM.dy * quiverX, ...
         quiverScale, 'w', 'LineWidth', 1.5)
  quiver(thisVM.x(thisVM.iReplacedCum), thisVM.y(thisVM.iReplacedCum), ...
         thisVM.dx(thisVM.iReplacedCum) * quiverX, ...
         thisVM.dy(thisVM.iReplacedCum) * quiverX, ...
         quiverScale, 'r', 'LineWidth', 1.5)
%   quiver(VM.x(thisVM.iReplaced), VM.y(thisVM.iReplaced), ...
%          VM.dx(thisVM.iReplaced) * quiverX, ...
%          VM.dy(thisVM.iReplaced) * quiverX, ...
%          quiverScale, 'r', 'LineWidth', 1.5)
  xlim([1 img_w]); ylim([1 img_h])
  %xlim([248 618]); ylim([262 538])
  title(thisVM.descr)
  %figure(1); xlim(round([248 618]/32)); ylim(round([262 538]/32))
  %figure(2); xlim(round([248 618]/16)); ylim(round([262 538]/16))
end

%% -----------------------------------------------------------------------
%   Show velocity component
% ------------------------------------------------------------------------

if 0
  %thisVM = VM;
  thisVM = VM_filt;
  
  % Do not plot zero magnitude vectors
  thisVM.x(isnan(thisVM.dx)) = nan;
  thisVM.y(isnan(thisVM.dy)) = nan;

  % Turn velocity component map into rgb image  
  [px_x, px_y] = meshgrid(1:img_w, 1:img_h);
  
  comp_nan2zero = thisVM.dx;
  comp_nan2zero(isnan(comp_nan2zero)) = 0;
  
  % Input grid [.x, .y] must not contain NaNs! Hence, use original IW_grid.
  img_comp = interp2(IW_grid_A.x, IW_grid_A.y, comp_nan2zero, px_x, px_y, ...
                     'nearest', 0);
  img_comp  = img_comp / max(abs(img_comp(:))) * 128 + 127;
  img_comp  = uint8(img_comp);
  img_comp  = ind2rgb(img_comp, brewermap(256, 'RdBu'));
  img_blend = ind2rgb(imadjust(A), flipud(gray(256))) .* img_comp;
  %img_blend = imfuse(img_comp, ind2rgb(imadjust(A), flipud(gray(256))), ...
  %                   'blend'); %, 'Scaling', 'independent');
  
  h7 = figure(7); clf
  set(gcf, 'Position', [2580, -225, 1280, 900])
  imshow(img_blend, 'InitialMagnification', 'fit')
  %imshow(img_comp, 'InitialMagnification', 'fit')
  colormap(brewermap(256, 'RdBu'))
  set(gca, 'Position', [0.06 0.05 .9 .91])
  axis on; axis tight; hold on
  
  %%
  quiver(thisVM.x, thisVM.y, thisVM.dx * quiverX, thisVM.dy * quiverX, ...
         quiverScale, 'k', 'LineWidth', 1.5)
  quiver(thisVM.x(thisVM.iReplacedCum), thisVM.y(thisVM.iReplacedCum), ...
         thisVM.dx(thisVM.iReplacedCum) * quiverX, ...
         thisVM.dy(thisVM.iReplacedCum) * quiverX, ...
         quiverScale, 'r', 'LineWidth', 1.5)
%   quiver(VM.x(thisVM.iReplaced), VM.y(thisVM.iReplaced), ...
%          VM.dx(thisVM.iReplaced) * quiverX, ...
%          VM.dy(thisVM.iReplaced) * quiverX, ...
%          quiverScale, 'r', 'LineWidth', 1.5)
  xlim([1 img_w]); ylim([1 img_h])
  %xlim([248 618]); ylim([262 538])
  title(thisVM.descr)
  %figure(1); xlim(round([248 618]/32)); ylim(round([262 538]/32))
  %figure(2); xlim(round([248 618]/16)); ylim(round([262 538]/16))
end


%% -----------------------------------------------------------------------
%   Individually colored vectors
% ------------------------------------------------------------------------

if 0
  % Show original full figure A
  h8 = figure(8); clf
  set(gcf, 'Position', [60, 60, 500, 500])
  imshow(imadjust(A), 'InitialMagnification', 80, 'Colormap', bone(256));
  hold on
  tmp = get(gcf, 'Position');
  set(gcf, 'Position', [2580, -225, tmp(3), tmp(4)])
  
  for iVector = 1:numel(VM.dx)
    vel_mag_color = cm(vel_mag_uint8(iVector) + 1, :);

    quiver(VM_filt.x(iVector), VM_filt.y(iVector), ...
           VM_filt.dx(iVector) * quiver_X, ...
           VM_filt.dy(iVector) * quiver_X, ...
           quiver_scale, 'LineWidth', 1.5, ...
           'Color', vel_mag_color, ...
           'MarkerFaceColor', vel_mag_color)
  end
  set(gca, 'YDir', 'reverse')
  axis tight; axis square
end

if fPROFILER
  profile off
  profile viewer
end