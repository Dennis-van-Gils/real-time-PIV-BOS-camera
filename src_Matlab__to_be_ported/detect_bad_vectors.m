function [iBadVectorAngle, iBadVectorMagn] = detect_bad_vectors(VM, ...
  angleDiffThreshold, magnDiffThreshold)

  fDEBUG = 0;
  DEBUG.iCOL = 7;

  % ----------------------------------------------------------------------
  %   Detect bad vectors, LP filter angle
  % ----------------------------------------------------------------------
  %angleDiffThreshold = 30;  % [degrees] leave at 30?
  % TODO: make the anglediffthreshold dependent on the grid size
  % Large grid ~ 128 px, use value 40
  % Small grid, go towards 30 or 20

  iBadVectorAngle = [];
  for iCol = 1:size(VM.x, 2)
    y = unwrap(VM.angle(:, iCol))/pi*180;

    % Split vector according to nans
    cut = isnan(y);
    id = cumsum(cut) + 1;
    mask = cut==0;
    out = accumarray(id(mask), y(mask), [], @(x) {x});
    offsets = find(cut == 1);

    for iSeg = 1:length(out)
      y = out{iSeg};

      if length(y) < 5
        % Do not perform outlier detection when segment is too short
        continue
      end

      if iSeg == 1
        y_offset = 0;
      else
        y_offset = offsets(iSeg - 1);
      end

      y_tiled = [flipud(y); y; flipud(y)];

      a = .5;
      y_LP = filtfilt(a, [1 a - 1], y_tiled);
      y_LP = y_LP(length(y) + 1:length(y) * 2);

      iR1 = find(abs((y - y_LP)) > angleDiffThreshold);
      iR1 = iR1 + y_offset;
      iR1 = sub2ind(size(VM.x), iR1, ones(size(iR1)) * iCol);
      
      if fDEBUG && iCol == DEBUG.iCol
        figure(10); clf
        plot(y   , 'x-k', 'Linewidth', 2); hold on
        plot(y_LP, 'ob' , 'Linewidth', 2)
        plot(abs(y - y_LP), 'x-g', 'Linewidth', 2)
        plot(xlim, [angleDiffThreshold angleDiffThreshold], '-k')
        title(['column = ' num2str(iCol) ', px\_x = ' num2str(VM.x(1, iCol))])
        xlabel('px\_y')
        drawnow
        pause
      end
      
      iBadVectorAngle = [iBadVectorAngle; iR1];                               %#ok<*AGROW>
    end
  end

  % ----------------------------------------------------------------------
  %  Detect bad vectors, LP filter magnitude
  % ----------------------------------------------------------------------
  %magnDiffThreshold = 4; 

  iBadVectorMagn = [];
  for iCol = 1:size(VM.x, 2)
    y = VM.magn(:, iCol);

    % Split vector according to nans
    cut = isnan(y);
    id = cumsum(cut) + 1;
    mask = cut==0;
    out = accumarray(id(mask), y(mask), [], @(x) {x});
    offsets = find(cut == 1);

    for iSeg = 1:length(out)
      y = out{iSeg};

      if length(y) < 5
        % Do not perform outlier detection when segment is too short
        continue
      end

      if iSeg == 1
        y_offset = 0;
      else
        y_offset = offsets(iSeg - 1);
      end

      y_tiled = [flipud(y); y; flipud(y)];

      a = .5;
      y_LP = filtfilt(a, [1 a - 1], y_tiled);
      y_LP = y_LP(length(y) + 1:length(y) * 2);

      iR1 = find(abs((y - y_LP)) > magnDiffThreshold);
      iR1 = iR1 + y_offset;
      iR1 = sub2ind(size(VM.x), iR1, ones(size(iR1)) * iCol);
      iBadVectorMagn = [iBadVectorMagn; iR1];
    end
  end

  if 1
    fprintf('nBadVectorAngle = %.0f\n', length(iBadVectorAngle))
    fprintf('iBadVectorMagn  = %.0f\n', length(iBadVectorMagn))
  end
end