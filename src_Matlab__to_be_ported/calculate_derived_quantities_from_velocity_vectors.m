function VM = calculate_derived_quantities_from_velocity_vectors(VM)
  % Dennis van Gils
  % 29-02-2016
  
  VM.magn  = sqrt(VM.dx.^2 + VM.dy.^2); % velocity magnitude  [px/frame]
  VM.angle = atan2(VM.dy, VM.dx);       % vector angle        [rad]
  VM.angle(VM.magn == 0) = nan;         % set undefined angle to nan
end