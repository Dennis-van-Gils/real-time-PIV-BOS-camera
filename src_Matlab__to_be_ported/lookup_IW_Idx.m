function iIW = lookup_IW_Idx(IW_grid, x_pixel, y_pixel)
  % Lookup the index of the IW that has it's center closest to the input
  % location [x_pixel, y_pixel]
  %
  % Output arguments:
  %   iIW: linear index of the IW
  %
  % Dennis van Gils
  % 23-02-2016

  iIW_x = floor((x_pixel - floor(IW_grid.IW_size/2) - 1) / ...
               (IW_grid.IW_size*(1 - IW_grid.overlap)) + 1.5);
  iIW_y = floor((y_pixel - floor(IW_grid.IW_size/2) - 1) / ...
               (IW_grid.IW_size*(1 - IW_grid.overlap)) + 1.5);
  iIW_x = min(iIW_x, IW_grid.nIWs_x);
  iIW_y = min(iIW_y, IW_grid.nIWs_y);
  iIW   = sub2ind([IW_grid.nIWs_y IW_grid.nIWs_x], iIW_y, iIW_x);

  %fprintf('%i, %i, %i\n', iIW, iIW_x, iIW_y)
end