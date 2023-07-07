function [px_sub, py_sub] = subpx_3pgf_2D(C, px, py)
  % Perform a 3-point Gaussian fit to the point with index (py, px)
  % inside of 2-D matrix 'C' along both the x and y direction.
  %
  % Dennis van Gils
  % 26-02-2016

  
  % Along x
  if px > 1 && px < size(C, 2) - 1
    % Fit possible
    phi_m1 = max(C(py, px - 1), 1e-40);   % Prevent taking log of zero
    phi_p1 = max(C(py, px + 1), 1e-40);   % Prevent taking log of zero
    px_sub = px + (log(phi_m1) - log(phi_p1)) / ...
                  (log(phi_m1) + log(phi_p1) - 2 * log(C(py, px))) / 2;
  else
    % No fit possible
    px_sub = px;
  end
  
  % Along y
  if py > 1 && py < size(C, 1) - 1
    % Fit possible
    phi_m1 = max(C(py - 1, px), 1e-40);   % Prevent taking log of zero
    phi_p1 = max(C(py + 1, px), 1e-40);   % Prevent taking log of zero
    py_sub = py + (log(phi_m1) - log(phi_p1)) / ...
                  (log(phi_m1) + log(phi_p1) - 2 * log(C(py, px))) / 2;
  else
    % No fit possible
    py_sub = py;
  end
end