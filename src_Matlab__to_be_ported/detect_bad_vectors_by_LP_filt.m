function VM_filt = detect_bad_vectors_by_LP_filt(VM, ...
  source_2D_matrix_to_LP, outlierThreshold)
  % Acts only column wise

  fDEBUG = 0;
  DEBUG.iCol = 7;

  iReplaced = [];
  for iCol = 1:size(VM.x, 2)
    y = source_2D_matrix_to_LP(:, iCol);

    % Split vector according to nans
    cut  = isnan(y);
    id   = cumsum(cut) + 1;
    mask = cut==0;
    out  = accumarray(id(mask), y(mask), [], @(x) {x});
    offsets = find(cut == 1);

    for iSeg = 1:length(out)
      y = out{iSeg};

      if length(y) < 5
        % Do not perform LP filter when segment is too short
        continue
      end

      if iSeg == 1
        y_offset = 0;
      else
        y_offset = offsets(iSeg - 1);
      end

      y_tiled = [flipud(y); y; flipud(y)];

      a = .5;   % LP filter -3dB (?, should check) point, 0.5 works fine
      y_LP = filtfilt(a, [1 a - 1], y_tiled);
      y_LP = y_LP(length(y) + 1:length(y) * 2);

      iR1 = find(abs((y - y_LP)) > outlierThreshold);
      iR1 = iR1 + y_offset;
      iR1 = sub2ind(size(VM.x), iR1, ones(size(iR1)) * iCol);
      
      if fDEBUG && iCol == DEBUG.iCol
        figure(10); clf
        plot(y   , 'x-k', 'Linewidth', 2); hold on
        plot(y_LP, 'ob' , 'Linewidth', 2)
        plot(abs(y - y_LP), 'x-g', 'Linewidth', 2)
        plot(xlim, [outlierThreshold outlierThreshold], '-k')
        title(['column = ' num2str(iCol) ...
               ', px\_x = ' num2str(VM.x(1, iCol))])
        xlabel('px\_y')
        drawnow
        pause
      end
      
      iReplaced = [iReplaced; iR1];                                         %#ok<*AGROW>
    end
  end
  
  VM_filt = VM;
  VM_filt.descr = 'filtered';
  VM_filt.dx(iReplaced)   = nan;
  VM_filt.dy(iReplaced)   = nan;
  VM_filt.iReplaced       = iReplaced;
  VM_filt.nReplaced       = length(iReplaced);
  VM_filt.iReplacedCum    = unique([VM_filt.iReplacedCum; iReplaced]);
  VM_filt.nReplacedCum    = length(VM_filt.iReplacedCum);

  % Recalculate derived quantities from velocity vectors
  VM_filt = calculate_derived_quantities_from_velocity_vectors(VM_filt);
end