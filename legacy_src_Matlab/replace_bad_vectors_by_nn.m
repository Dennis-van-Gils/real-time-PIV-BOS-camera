function VM_filt = replace_bad_vectors_by_nn(VM_filt)
  % Replace bad vectors by nearest neighborhood
  %
  % Dennis van Gils
  % 28-02-2016

  for iNr = 1:VM_filt.nReplaced
    iIW = VM_filt.iReplaced(iNr);
    [iIW_y, iIW_x] = ind2sub(size(VM_filt.dx), iIW);

    d = [ 1 0; -1 0; 1 1; 0 1; -1 1; 1 -1; 0 -1; -1 -1];
    neighbors = d + repmat([iIW_y, iIW_x], [8 1]);
    neighbors(neighbors(:, 1) < 1, :) = [];
    neighbors(neighbors(:, 2) < 1, :) = [];
    neighbors(neighbors(:, 1) > size(VM_filt.x, 1), :) = [];
    neighbors(neighbors(:, 2) > size(VM_filt.x, 2), :) = [];
    neighbors = sub2ind(size(VM_filt.dx), neighbors(:, 1), neighbors(:, 2));

    VM_filt.dx(iIW) = nanmedian(VM_filt.dx(neighbors));
    VM_filt.dy(iIW) = nanmedian(VM_filt.dy(neighbors));
  end

  % Recalculate derived quantities
  VM_filt = calculate_derived_quantities_from_velocity_vectors(VM_filt);
end