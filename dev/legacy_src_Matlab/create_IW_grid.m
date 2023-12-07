function IW_grid = create_IW_grid(img_w, img_h, IW_size, overlap)
  % Input arguments:
  %   img_w  : width of source image A      [px]
  %   img_h  : height of source image A     [px]
  %   IW_size: interrogation window size    [px]
  %   overlap factor:                       [0-1]
  %     0  : no window overlap
  %     0.5: 50% window overlap
  %
  % Output arguments:
  %   IWs                                   [structure]
  %     .IW_size : direct copy of input parameter 'IW_size'
  %     .overlap : direct copy of input parameter 'overlap'
  %     .nIWs_x  : number of interrogation windows along the x direction
  %     .nIWs_y  : number of interrogation windows along the y direction
  %     .nIWs    : total number of interrogation windows
  %     .x       : meshgrid of the x-positions of the IW centers  [px]
  %     .y       : meshgrid of the y-positions of the IW centers  [px]
  %     .x_range : array containing [min max] x-pos per IW        [px]
  %     .y_range : array containing [min max] y-pos per IW        [px]
  %
  % Dennis van Gils
  % 24-02-2016

  % Store the input arguments in the output structure
  IW_grid.IW_size = IW_size;
  IW_grid.overlap = overlap;

  % Calculate number of IWs that will fit in the source image
  nIWs_x = floor((img_w - IW_size)/IW_size/(1 - overlap)) + 1;
  nIWs_y = floor((img_h - IW_size)/IW_size/(1 - overlap)) + 1;
  IW_grid.nIWs_x = nIWs_x;
  IW_grid.nIWs_y = nIWs_y;
  IW_grid.nIWs   = nIWs_x * nIWs_y;

  % Calculate IW positions
  array_x = round((0:nIWs_x - 1) * (1 - overlap) * IW_size + ...
                  floor(IW_size/2) + 1);
  array_y = round((0:nIWs_y - 1) * (1 - overlap) * IW_size + ...
                  floor(IW_size/2) + 1);
  [IW_grid.x, IW_grid.y] = meshgrid(array_x, array_y);

  % Calculate IW ranges
  x_range = [array_x - floor(IW_size/2); array_x + floor(IW_size/2) - 1]';
  y_range = [array_y - floor(IW_size/2); array_y + floor(IW_size/2) - 1]';
  IW_grid.x_range = sortrows(repmat(x_range, nIWs_y, 1));
  IW_grid.y_range = repmat(y_range, nIWs_x, 1);

  fDEBUG = 0;
  if fDEBUG
    figure(1); clf
    plot(IW_grid.x_range(:, 1), IW_grid.y_range(:, 1), 'xg', ...
         'Linewidth', 2, 'DisplayName', 'IW starts')
    hold on
    plot(IW_grid.x_range(:, 2), IW_grid.y_range(:, 2), 'xr', ...
         'Linewidth', 2, 'DisplayName', 'IW endings')
    plot(IW_grid.x(:, 1), IW_grid.y(:, 1), 'ok', ...
         'DisplayName', 'IW centers')
    plot(IW_grid.x(1, :), IW_grid.y(1, :), 'ok', 'HandleVisibility', 'off')
    plot([img_w img_w], [1 img_h], '-', 'HandleVisibility', 'off')
    plot([1 img_w], [img_h img_h], '-', 'HandleVisibility', 'off')
    legend('Location', 'NO', 'Orientation', 'horizontal')
  end
end