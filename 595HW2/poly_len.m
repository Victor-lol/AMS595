function L = poly_len(p, s, e)
% POLY_LEN  Arc length of polynomial y(x)=polyval(p,x) over [s,e].
% INPUTS: p (poly coeffs as from polyfit), s,e bounds
% OUTPUT: L (scalar length)
%
% Formula: L = ∫_s^e sqrt(1 + (y'(x))^2) dx
% Use polyder + integral (MATLAB) or quadgk/quad (Octave).

    dp = polyder(p);                 % derivative coefficients
    ds = @(x) sqrt(1 + (polyval(dp,x).^2));
    % Safer integrator choice in MATLAB:
    L = integral(ds, s, e, 'AbsTol',1e-8, 'RelTol',1e-8);
end