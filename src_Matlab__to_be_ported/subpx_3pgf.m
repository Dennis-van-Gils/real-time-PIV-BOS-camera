function px_sub = subpx_3pgf(C, px)
  % Perform a 3-point Gaussian fit to the point with index 'px' inside of
  % array 'C'.
  %
  % Dennis van Gils
  % 26-02-2016

  if px > 1 && px < length(C) - 1
    % Fit possible
    phi_m1 = max(C(px - 1), 1e-40);   % Prevent taking log of zero
    phi_p1 = max(C(px + 1), 1e-40);   % Prevent taking log of zero
    px_sub = px + (log(phi_m1) - log(phi_p1)) / ...
                  (log(phi_m1) + log(phi_p1) - 2 * log(C(px))) / 2;
  else
    % No fit possible
    px_sub = px;
  end
end