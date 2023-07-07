function VM_filt = detect_bad_vectors_by_MAD(VM, MAD_threshold)
  % Filter vectors based on the median absolute deviation of the magnitude
  % TO DO: keep track of applied filter history inside of VM structure
  %
  % Input arguments:
  %   outlier_threshold
  %     < ~1    : is low, lots of good vectors injustly marked as bad
  %     2 to 2.5: right balance
  %     4       : is high, only the worst vectors are marked as bad
  %
  % Dennis van Gils
  % 29-02-2016

  c = (VM.magn(:) - nanmedian(VM.magn(:))) / mad(VM.magn(:));
  iReplaced = find(c > (nanmean(abs(c)) + ...
                        nanstd(abs(c)) * MAD_threshold));
  nReplaced = length(iReplaced);

  VM_filt = VM;
  VM_filt.descr = 'filtered';
  VM_filt.dx(iReplaced)   = nan;
  VM_filt.dy(iReplaced)   = nan;
  VM_filt.iReplaced       = iReplaced;
  VM_filt.nReplaced       = nReplaced;
  VM_filt.iReplacedCum    = unique([VM_filt.iReplacedCum; iReplaced]);
  VM_filt.nReplacedCum    = length(VM_filt.iReplacedCum);

  % Recalculate derived quantities from velocity vectors
  VM_filt = calculate_derived_quantities_from_velocity_vectors(VM_filt);
end