function y = snr_xuf(x0, x)
  mse = norm(x0 - x,'fro');
  y = norm(x0, 'fro');

  y = 20*log10(y/(mse + 1e-30));
end
