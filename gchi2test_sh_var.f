      real prob,  efer(11),ffer(11),fereru(11),fererl(11),
     *            echer(17),fcher(17),fcheru(17),fcherl(17),
     *            emod(150),fmod(150),
     *   ee(30),fe(30),feer(30),  aa,bb,  efl(11),efr(11),  chi2, p, co,
     *      fmodee(30), fmco(30),     sum1,sum2,de(30),  ecl(17),ecr(17)
      integer ne,  i,j,k,  nf,nc, mm,   l, lmin
      real shift, shiftopt, chi2min
      data chi2min/1000000.0/
**      data shift/1.2/
**      data shift/0.8/
**      data shift/1.0/
**      data shift/0.85/
**      data shift/0.75/
**      data shift/0.95/
**      data shift/1.05/


                        do l=1,31   ! loop in shift begins
			  shift = 1.01 - 0.01*l

      write(*,*) ''
      write(*,*) '    shift=',shift
      write(*,*) ''

      open(unit=10,file='SED-KD10-Basic-0.140-Combined_1ES0229+200',
     *     status='OLD',form='FORMATTED')
      do i=1,150
        read(10,*) emod(i),fmod(i)
      enddo
      close(10)

      open(unit=10,file='1ES0229+200-Fermi',
     *     status='OLD',form='FORMATTED')
      read(10,*) nf
      do i=1,nf
        read(10,*) efl(i),efer(i),efr(i),ffer(i),fererl(i),fereru(i)
	efl(i) = efl(i)*1.0e-6             ! left energy
	efer(i) = efer(i)*1.0e-6           ! mid energy
	efr(i) = efr(i)*1.0e-6             ! right energy
	ffer(i) = ffer(i)*1.0e-6           ! SED
	fererl(i) = fererl(i)*1.0e-6       ! SED lower error
	fereru(i) = fereru(i)*1.0e-6       ! SED upper error
      enddo
      close(10)
      write(*,*) ''
      write(*,*) '    nf=',nf
      write(*,*) ''
      write(*,*) '     efer:'
      write(*,100) (efer(i),i=1,nf)
      write(*,*) ''
      write(*,*) '     ffer:'
      write(*,100) (ffer(i),i=1,nf)
      write(*,*) ''
      write(*,*) '     fererl:'
      write(*,100) (fererl(i),i=1,nf)
      write(*,*) ''
      write(*,*) '     fereru:'
      write(*,100) (fereru(i),i=1,nf)
      write(*,*) ''

      open(unit=10,file='1ES0229+200-IACT',
     *     status='OLD',form='FORMATTED')
      read(10,*) nc,aa
      do i=1,nc
        read(10,*) mm,echer(i),ecl(i),ecr(i),fcher(i),fcheru(i),
**        read(10,*) echer(i),ecl(i),ecr(i),fcher(i),fcheru(i),
     *             fcherl(i),aa,bb
        echer(i) = echer(i)*shift                  ! mid energy
	fcher(i) = fcher(i)*echer(i)*echer(i)      ! SED
	fcheru(i) = fcheru(i)*echer(i)*echer(i)    ! SED upper error
	fcherl(i) = -fcherl(i)*echer(i)*echer(i)   ! SED lower error
      enddo
      close(10)
      write(*,*) ''
      write(*,*) '    nc=',nc
      write(*,*) ''

**  define energy grid for the test and model SEDs for the grid

      j = 0
**      do i=1,30
      do i=1,nf+nc
        if(i.le.nf) then      ! Fermi data
	  if(fererl(i).gt.0.0) then    ! consider only reasonable data
	    j = j + 1
***      write(*,*) '  ** j=',j,'  i=',i
	    ee(j) = efer(i)
	    fe(j) = ffer(i)
	    feer(j) = 0.5*(fererl(i)+fereru(i))
	    de(j) = efr(i) - efl(i)
	  endif
        else                 ! cher data
	  j = j + 1
***      write(*,*) '  **** j=',j,'  i=',i
	  ee(j) = echer(i-nf)
	  fe(j) = fcher(i-nf)
	  feer(j) = 0.5*(fcherl(i-nf)+fcheru(i-nf))
	  de(j) = ecr(i-nf) - ecl(i-nf)
	endif
      enddo
      write(*,*) ''
      write(*,*) '      j=',j    !  number of selected points
      write(*,*) ''
      write(*,*) '         ee:'
      write(*,100) (ee(i),i=1,j)
      write(*,*) ''
      write(*,*) '         fe:'
      write(*,100) (fe(i),i=1,j)
      write(*,*) ''
      write(*,*) '         feer:'
      write(*,100) (feer(i),i=1,j)
      write(*,*) ''
      write(*,*) '         de:'
      write(*,100) (de(i),i=1,j)
      write(*,*) ''

      do i=1,j
        do k=1,150
	  if(emod(k).gt.ee(i)) then
	    fmodee(i) = fmod(k-1) + (fmod(k)-fmod(k-1))
     *                 /(emod(k)-emod(k-1))*(ee(i)-emod(k-1))
**      write(*,*) '    *  i=',i,' ee(i)=',ee(i),' k=',k,
**     *            ' fmodee(i)=',fmodee(i)
            go to 10
	  endif
	enddo
   10	continue
      enddo
      write(*,*) ''
      write(*,*) '   model points adjusted to exp. energy grid'
      write(*,*) ''
      write(*,*) '         ee:'
      write(*,100) (ee(i),i=1,j)
      write(*,*) ''
      write(*,*) '         fmodee:'
      write(*,100) (fmodee(i),i=1,j)
      write(*,*) ''
      write(*,*) ''
      write(*,*) ''
      write(*,*) ''


*******   apply chi2-test

            do k=1,16

      co = 0.90**(k-1)*2.5
***      co = 0.95**(k-1)
                write(*,*) '       k=',k,
     *                     '  model SED is multipied by ',co
      chi2 = 0.0
      do i=1,j
        fmco(i) = fmodee(i)*co
**        chi2 = chi2 + ((fmodee(i) - fe(i))/feer(i))**2
        chi2 = chi2 + ((fmco(i) - fe(i))/feer(i))**2
***        chi2 = chi2 + ((fmco(i) - fe(i))/(3.0*feer(i)))**2
      enddo

            if(chi2min.gt.chi2) then
	      chi2min = chi2
	      lmin = l
	      shiftopt = shift
	    endif

      write(*,*) ''
      write(*,*) '       chi2=',chi2,' ndf=',j-1,' chi2/ndf=',chi2/(j-1)
      p = prob(chi2,j-1)
      write(*,*) ''
      write(*,*) '    p=',p
      write(*,*) ''
      write(*,*) ''

            enddo
      write(*,*) ''


                        enddo     ! loop in shift ends

      write(*,*) ''
      write(*,*) ''
      write(*,*) ''
      write(*,*) '       chi2min=',chi2min,' lmin=',lmin
      write(*,*) ''
      write(*,*) '        optimal shift=',shiftopt
                        shift = shiftopt
      write(*,*) ''
      write(*,*) '   Repeat the trial for the optimal shift'
      write(*,*) ''
      write(*,*) ''


      open(unit=10,file='1ES0229+200-IACT',
     *     status='OLD',form='FORMATTED')
      read(10,*) nc,aa
      do i=1,nc
        read(10,*) mm,echer(i),ecl(i),ecr(i),fcher(i),fcheru(i),
**        read(10,*) echer(i),ecl(i),ecr(i),fcher(i),fcheru(i),
     *             fcherl(i),aa,bb
        echer(i) = echer(i)*shift                  ! mid energy
	fcher(i) = fcher(i)*echer(i)*echer(i)      ! SED
	fcheru(i) = fcheru(i)*echer(i)*echer(i)    ! SED upper error
	fcherl(i) = -fcherl(i)*echer(i)*echer(i)   ! SED lower error
      enddo
      close(10)
      write(*,*) ''
      write(*,*) '    nc=',nc
      write(*,*) ''

**  define energy grid for the test and model SEDs for the grid

      j = 0
**      do i=1,30
      do i=1,nf+nc
        if(i.le.nf) then      ! Fermi data
	  if(fererl(i).gt.0.0) then    ! consider only reasonable data
	    j = j + 1
***      write(*,*) '  ** j=',j,'  i=',i
	    ee(j) = efer(i)
	    fe(j) = ffer(i)
	    feer(j) = 0.5*(fererl(i)+fereru(i))
	    de(j) = efr(i) - efl(i)
	  endif
        else                 ! cher data
	  j = j + 1
***      write(*,*) '  **** j=',j,'  i=',i
	  ee(j) = echer(i-nf)
	  fe(j) = fcher(i-nf)
	  feer(j) = 0.5*(fcherl(i-nf)+fcheru(i-nf))
	  de(j) = ecr(i-nf) - ecl(i-nf)
	endif
      enddo
      write(*,*) ''
      write(*,*) '      j=',j    !  number of selected points
      write(*,*) ''
      write(*,*) '         ee:'
      write(*,100) (ee(i),i=1,j)
      write(*,*) ''
      write(*,*) '         fe:'
      write(*,100) (fe(i),i=1,j)
      write(*,*) ''
      write(*,*) '         feer:'
      write(*,100) (feer(i),i=1,j)
      write(*,*) ''
      write(*,*) '         de:'
      write(*,100) (de(i),i=1,j)
      write(*,*) ''

      do i=1,j
        do k=1,150
	  if(emod(k).gt.ee(i)) then
	    fmodee(i) = fmod(k-1) + (fmod(k)-fmod(k-1))
     *                 /(emod(k)-emod(k-1))*(ee(i)-emod(k-1))
**      write(*,*) '    *  i=',i,' ee(i)=',ee(i),' k=',k,
**     *            ' fmodee(i)=',fmodee(i)
            go to 20
	  endif
	enddo
   20	continue
      enddo
      write(*,*) ''
      write(*,*) '   model points adjusted to exp. energy grid'
      write(*,*) ''
      write(*,*) '         ee:'
      write(*,100) (ee(i),i=1,j)
      write(*,*) ''
      write(*,*) '         fmodee:'
      write(*,100) (fmodee(i),i=1,j)
      write(*,*) ''
      write(*,*) ''
      write(*,*) ''
      write(*,*) ''


*******   apply chi2-test

            do k=1,16

      co = 0.90**(k-1)*2.5
***      co = 0.95**(k-1)
                write(*,*) '       k=',k,
     *                     '  model SED is multipied by ',co
      chi2 = 0.0
      do i=1,j
        fmco(i) = fmodee(i)*co
**        chi2 = chi2 + ((fmodee(i) - fe(i))/feer(i))**2
        chi2 = chi2 + ((fmco(i) - fe(i))/feer(i))**2
***        chi2 = chi2 + ((fmco(i) - fe(i))/(3.0*feer(i)))**2
      enddo

            if(chi2min.gt.chi2) then
	      chi2min = chi2
	      lmin = l
	    endif

      write(*,*) ''
      write(*,*) '       chi2=',chi2,' ndf=',j-1,' chi2/ndf=',chi2/(j-1)
      p = prob(chi2,j-1)
      write(*,*) ''
      write(*,*) '    p=',p
      write(*,*) ''
      write(*,*) ''

            enddo
      write(*,*) ''


***      chi2 = 10.0
**      j = 20
***      write(*,*) '       chi2=',chi2,' ndf=',j-1,' chi2/ndf=',chi2/(j-1)
***      p = prob(chi2,j-1)
***      write(*,*) '    p=',p

      write(*,*) ''
      write(*,*) ''
      write(*,*) ''
      write(*,*) ''
      write(*,*) '   trying to account for normalization difference ...'
      write(*,*) ''

      sum1 = 0.0
      sum2 = 0.0
      do i=1,j
        sum1 = sum1 + de(i)*fe(i)
        sum2 = sum2 + de(i)*fmodee(i)
      enddo
      co = sum1/sum2
      write(*,*) '    sum1=',sum1,'  sum2=',sum2,'  sum1/sum2=',
     *                  sum1/sum2
      write(*,*) ''

                write(*,*) '       ',
     *                     '  model SED is multipied by ',co
      chi2 = 0.0
      do i=1,j
        fmco(i) = fmodee(i)*co
**        chi2 = chi2 + ((fmodee(i) - fe(i))/feer(i))**2
        chi2 = chi2 + ((fmco(i) - fe(i))/feer(i))**2
***        chi2 = chi2 + ((fmco(i) - fe(i))/(3.0*feer(i)))**2
      enddo
      write(*,*) ''
      write(*,*) '       chi2=',chi2,' ndf=',j-1,' chi2/ndf=',chi2/(j-1)
      p = prob(chi2,j-1)
      write(*,*) ''
      write(*,*) '    p=',p
      write(*,*) ''
      write(*,*) ''

      open(unit=20,file='gchi2test_sh_var_out',status='NEW',
     *       form='FORMATTED')
      do i=1,j
        write(20,200) ee(i),fe(i),feer(i),fmodee(i),fmco(i)
      enddo
      close(20)


      stop
  100 format(10(1x,1pe10.3))
  200 format(5(1x,1pe12.5))
      end

*************************************************************************************

      FUNCTION PROB(X,N)
 
**#include "gen/imp64.inc"
      REAL PROB,X
      CHARACTER NAME*(*)
      CHARACTER*80 ERRTXT
      PARAMETER (NAME = 'PROB')
      PARAMETER (R1 = 1, HF = R1/2, TH = R1/3, F1 = 2*R1/9)
      PARAMETER (C1 = 1.12837 91670 95513D0)
      PARAMETER (NMAX = 300)
*                maximum chi2 per df for df >= 2., if chi2/df > chipdf prob=0.
      PARAMETER (CHIPDF = 100.)
      PARAMETER (XMAX = 174.673, XMAX2 = 2*XMAX)
**#if defined(CERNLIB_IBM)
*
*     13.3 is limit of DERFC intrinsic (Wojciech Wojcik/IN2P3)
*
**      PARAMETER (XLIM = 13.3)
**#endif
**#if !defined(CERNLIB_IBM)
      PARAMETER (XLIM = 24.)
**#endif
**      PARAMETER (EPS = 1D-30)
      PARAMETER (EPS = 1D-10)
**#if defined(CERNLIB_DOUBLE)
**      GERFC(V)=DERFC(V)
**#endif
**#if !defined(CERNLIB_DOUBLE)
****************************      GERFC(V)= ERFC(V)
**#endif

****      real*8 y,u,h,w,n,s,t,fi

 
      Y=X
      U=HF*Y
      IF(N .LE. 0) THEN
       H=0
       WRITE(*,101) N
**       CALL MTLPRT(NAME,'G100.1',ERRTXT)
      ELSEIF(Y .LT. 0) THEN
       H=0
        WRITE(*,102) X
**       CALL MTLPRT(NAME,'G100.2',ERRTXT)
      ELSEIF(Y .EQ. 0 .OR. N/20 .GT. Y) THEN
       H=1
      ELSEIF(N .EQ. 1) THEN
       W=SQRT(U)
       IF(W .LT. XLIM) THEN
**        H=GERFC(W)
        H=ERFC(W)
       ELSE
        H=0
       ENDIF
      ELSEIF(N .GT. NMAX) THEN
       S=R1/N
       T=F1*S
       W=((Y*S)**TH-(1-T))/SQRT(2*T)
       IF(W .LT. -XLIM) THEN
        H=1
       ELSEIF(W .LT. XLIM) THEN
**        H=HF*GERFC(W)
        H=HF*ERFC(W)
       ELSE
        H=0
       ENDIF
      ELSE
       M=N/2
       IF(U .LT. XMAX2 .AND. (Y/N).LE.CHIPDF ) THEN
        S=EXP(-HF*U)
        T=S
        E=S
        IF(2*M .EQ. N) THEN
         FI=0
         DO I = 1,M-1
           FI=FI+1
           T=U*T/FI
    1      S=S+T
         enddo
         H=S*E
        ELSE
         FI=1
         DO I=1,M-1
           FI=FI+2
           T=T*Y/FI
    2      S=S+T
         enddo
         W=SQRT(U)
         IF(W.LT.XLIM) THEN
**          H=C1*W*S*E+GERFC(W)
          H=C1*W*S*E+ERFC(W)
         ELSE
          H=0.
         ENDIF
        ENDIF
       ELSE
        H=0
       ENDIF
      ENDIF
      IF ( H.GT. EPS ) THEN
         PROB=H
**         PROB=sngl(H)
      ELSE
         PROB=0.
      ENDIF
      RETURN
  101 FORMAT('N = ',I6,' < 1')
  102 FORMAT('X = ',1P,E20.10,' < 0')
      END