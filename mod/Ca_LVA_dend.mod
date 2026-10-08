:Comment : Copy of Ca_LVA.mod for the dendrite, with an optional shift of the ACTIVATION gate only.
:          Added for the PV_dendrites project; everything else is identical to Ca_LVA.mod.
:
:          Current:  ica = gbar * m^2 * h * (v - eca)
:          Gates:    dm/dt = (mInf - m)/mTau,  dh/dt = (hInf - h)/hTau
:          The activation gate (mInf, mTau) is evaluated at v - vshift_act, so a negative vshift_act
:          makes the channel open at more negative potentials (vshift_act = -15 mV moves the
:          half-activation of m from -40 to -55 mV). The inactivation gate (hInf, hTau) is unchanged.
:          With vshift_act = 0 this mechanism reproduces Ca_LVA.mod exactly (tested in tests/).
:          A separate SUFFIX is used so that the soma and axon keep the original Ca_LVA.
:Comment : LVA ca channel. Note: mtau is an approximation from the plots
:Reference : :		Avery and Johnston 1996, tau from Randall 1997
:Comment: shifted by -10 mv to correct for junction potential
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX Ca_LVA_dend
	USEION ca READ eca WRITE ica
	RANGE gbar, g, ica, vshift_act
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
	vshift_act = 0 (mV)	: shift of the activation gate (negative: activates earlier)
}

ASSIGNED	{
	v	(mV)
	eca	(mV)
	ica	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau
	hInf
	hTau
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar*m*m*h
	ica = g*(v-eca)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((34-21)/10)

	UNITSOFF
		: the original channel evaluates its rates at v + 10 mV (junction-potential correction);
		: here the activation gate is additionally evaluated at v - vshift_act
		v = v + 10 - vshift_act
		mInf = 1.0000/(1+ exp((v - -30.000)/-6))
		mTau = (5.0000 + 20.0000/(1+exp((v - -25.000)/5)))/qt
		v = v + vshift_act	: back to v + 10 for the (unshifted) inactivation gate
		hInf = 1.0000/(1+ exp((v - -80.000)/6.4))
		hTau = (20.0000 + 50.0000/(1+exp((v - -40.000)/7)))/qt
		v = v - 10
	UNITSON
}
