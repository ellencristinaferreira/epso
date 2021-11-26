%% Programa EPSO 06/09/2016
%% Funcionando com todos os sistemas
%% MODELO MULTI-OBJETIVO
%% Mnimizacao de perdas & minimização de reativos

clc
close all
clear all
format short
warning('off')

%% MODELOS
% MODELO CONTINUO TAP E SHUNT FIXOS = 1
% MODELO CONTINUO TAP E SHUNT CONTINUOS = 2
% MODELO TAP E SHUNT DISCRETOS = 3
% MODELO ARREDONDAMENTO = 4

%% MODELOS COM GRADIENTE
% MODELO CONTINUO TAP E SHUNT FIXOS COM GRADIENTE = 51
% MODELO CONTINUO TAP E SHUNT CONTINUOS COM GRADIENTE = 52
% MODELO TAP E SHUNT DISCRETOS COM GRADIENTE= 53
% MODELO ARREDONDAMENTO COM GRADIENTE= 54

%% MODELO MULTIOBJETIVO
% MODELO COM MARGEM DE REATIVOS CONTINUO TAP E SHUNT FIXOS = 61

modelo = 61; % modelo a ser executado


%% METAHEURISTICA
% META =   EPSO = 1
% META = DEEPSO = 21 (Sg: escolha fixa do vetor x - particulas)
% META = DEEPSO = 22 (Sg-rnd: escolha aleatoria do vetor x - particulas)
% META = DEEPSO = 23 (Pb: escolha fixa do vetor pbest )
% META = DEEPSO = 24 (Pb-rnd: escolha aleatoria do vetor Pbest)
meta = 1;


%% IMPRESSAO RELATORIO
% total = 0; imprime apenas as iteracoes
% total = 1; imprime todas as variaveis e restricoes
% if modelo==3 || modelo==53 || modelo==63
    rel_total = 1;
% else
%     rel_total = 0;
% end

%% SISTEMAS
sistem = 14;
[filename] = sistema(sistem);


%% parametro de ponderacao do modelo multiobjetivo W = funcao perdas;
% 1-W = gracao de reativos
% se W=0 minimiza apenas geracao de reativos
% se W=1 minimiza apenas funcao perdas
W = 0.5;

%% soma ponderada para cada sistema

if sistem == 14
    sum_min_loss = 13.36;
    sum_max_loss = 16.46;
    sum_min_reactive = 0.23945;
    sum_max_reactive = 1.901862;
end


%% ITERACOES E PARTICULAS DO EPSO
itermax = 100; % numero de iteracoes maxima
npart = 50; % numero de particulas do PSO
cont_grad_max = 5;
% if sistem ==300
%     cont_grad_max = 1;
% end

% PSO com GRADIENTE
%     itermax = 100; % numero de iteracoes maxima
%     npart = 10; % numero de particulas do PSO

    
%% parametros para calculo da velocidade para DEEPSO
sg = randi(npart,1,1); % escolha fixa da posicao do vetor particula
rr = randi(npart,1,1); %escolha fixa do vetor pbest


%% calculo do P para DEEPSO
sm = npart;
p = 0.75;
PP = DEEPSO_P(sm, p); % funcao que calcula o P


%% rotina que escolhe o modelo e o sistema para impressao do txt e xls
% [arquivo] = impressao(sistem,modelo,meta);


%% QUANTIDADE DE SEMENTES
yy=1; % inicializando a variavel
nseed = 1; % numero de sementes
fobj = zeros(nseed,1);
% tempo = zeros(nseed,1);
% t1 = clock;

for yy=1:nseed
    
    [s]=semente(yy);
    RandStream.setGlobalStream(s);
    %         RandStream.setDefaultStream(s);
    
    addpath('C:\Ellen\func');
    
    initpsat% inicializando psat
    clpsat.readfile = 0;% Nao recarregar arquivo de dados
    Settings.show = 0;% Desabilitar impressao de mensagens na tela
    clpsat.refresh = false;% atualizar se for TRUE (default), repetir a analise do fluxo de energia
    Settings.pv2pq = 1; % habilita os controles da geracao de potencia reativa
    
    % Abrindo os arquivos de dados do psat
    runpsat(filename,'C:\Ellen\psat\dados','data');
    
    %Executando fluxo de carga
    %=============================
    runpsat('pf');
    %=============================
    
    %% Inicializacoes
    %=============================
    o = Line.con(:,1);% barra origem
    d = Line.con(:,2);% barra destino
    gkm = Line.con(:,8)./(Line.con(:,8).^2 + Line.con(:,9).^2);% calculo de gkm
    bkm = -Line.con(:,9)./(Line.con(:,8).^2 + Line.con(:,9).^2);% calculo de bkm
    bkmsh = Line.con(:,10)./2; % bkmshunt
    akm = Line.con(:,11);% tap
    nb = Bus.n;% numero de barras
    tipo = zeros([nb,1]);% criando um vetor de tipos de barra com numeros zeros
    npv = size(PV.con,1);% numero de barras PV
    npq = nb - npv - 1;% numero de barras PQ
    nlin = Line.n;% numero de linhas
    Qmin = zeros(nb,1);
    Qmax = zeros(nb,1);
    
    
    %=============================
    % criando vetor tipo
    % Barra1=Vteta Barra2=PV Barra3=PQ
    % tipo=1 (PV) tipo=2(slack) tipo=0(PQ)
    % barra   tipo
    % 1         2 de referencia
    % 2         1
    % 3         0
    %=============================
    
    % criando barras do tipo PV = 1
    %=============================
    for i = 1:npv
        tipo(PV.con(i,1))=1;
    end
    
    % criando barras do tipo slack = 2
    %=============================
    for i = 1:size(SW.con,1)
        tipo(SW.con(i,1))=2;
    end
    %=============================
    
    % criando vetor Qmax e Qmin
    %=============================
    for i=1:nb
        if tipo(i) == 2
            Qmin(i)=SW.con(1,7);
        end
    end
    
    ii=1;
    for i=1:nb
        if tipo(i) == 1
            Qmin(i)=PV.con(ii,7);
            ii=ii+1;
        end
    end
    
    for i=1:nb
        if tipo(i) == 2
            Qmax(i)=SW.con(1,6);
        end
    end
    
    ii=1;
    for i=1:nb
        if tipo(i) == 1
            Qmax(i)=PV.con(ii,6);
            ii=ii+1;
        end
    end
    %=============================
    
    % criando o vetor b shunt
    %=============================
    bsh = zeros(nb,1);
    for i = 1:size(Shunt.con,1)
        bsh(Shunt.con(i,1)) = Shunt.con(i,6);
    end
    %=============================
    
    % atribuindo os valores da matriz Ybus
    %=============================
    G = real(Snapshot.Ybus);
    B = imag(Snapshot.Ybus);
    %=============================
    
    % criando vetor potencia gerada
    %=============================
    Pg = zeros(nb,1);
    for i=1:npv
        Pg(PV.con(i,1)) = PV.con(i,4);
    end
    %=============================
    
    % criando vetores de P e Q calculados
    %=============================
    Pc = zeros(nb,1);
    Qc = zeros(nb,1);
    for i=1:size(PQ.con,1)
        Pc(PQ.con(i,1)) = PQ.con(i,4);
        Qc(PQ.con(i,1)) = PQ.con(i,5);
    end
    Qg = zeros(nb,1);% criando o vetor Qgerado com zeros
    %=============================
    
    % variaveis V e teta recebem os valores vindo do FC do PSAT
    %=============================
    V = Snapshot.y(1+nb:2*nb);
    teta = Snapshot.y(1:nb);
    %=============================
    
    % especificando os limites de magnitude de tensao e taps
    %=============================
    Vmax = 1.1;
    Vmin = 0.9;
    tmax = 1.12;
    tmin = 0.88;
    apasso = 0.0075;
    %=============================
    
    % atribuindo os valores para a variavel tap vindo do psat
    %=============================
    i = 1;
    jj = 1;
    for i=1:nlin
        if Line.con(i,11)~= 1.00
            tap(jj)=Line.con(i,11);
            jj = jj + 1;
        end
    end
    %=============================
    
    %% =============== Inicio EPSO ===============
    
    nu = nb - npq;% vetor utilizando apenas as variaveis de controle
    na = size(tap,2); % apenas variaveis de controle TAP
    nsh = length(Shunt.bus);
    iter = 0; % contador iteracao PSO
    
    Vpb = zeros(npart,nb);% vetor pbest para todas variaveis tensao
    Apb = zeros(npart,na); % vetor pbest para variaveis tap
    Spb = zeros(npart,nsh);% vetor pbest para variavel shunt
    
    fopb = zeros(npart,1);% inicializando funcao pbest
    fo = zeros(npart,1); %inicializacao da funcao objetivo
    fperdas = zeros(npart,1);
    geracao = zeros(npart,1);
    reativo_gera = 0;
    
    penal1 = zeros(npart,1);
    penal2 = zeros(npart,1);
    penal3 = zeros(npart,1);
    
    deltap = zeros(npart,nb);
    deltaq = zeros(npart,nb);
    Qgpb = zeros(npart,nb);
    
    pbest = [Vpb Apb fopb deltap deltaq fperdas Qgpb Spb];
    pbest_new = zeros(npart,nb+na+1+nb+nb+1+nb+nsh);%
    gbest = zeros(1,nb+na+1+nb+nb+1+nb+nsh);% inicializando vetor gbest
    gbest_gauss = zeros(1,nb+na+nsh);
    gbestnew = 0;
    gbest0 = 100;
    
    
    cont = 0;% criterio de parada da metaheuristica
    
    cont_grad = 0;% contador do gradiente
    %=============================
    
    %% Inicializacao das variaveis de controle
    
    if modelo == 51 || modelo == 52 || modelo == 53 || modelo == 54
        % ======== Gradiente ========
        tol = 2*10^-3; % tolerancia usada no gradiente reduzido
        maxit = 100; % numero de iteracoes maxima dentro do gradiente
        it = 0; % contador de iteracoes do gradiente red.
        w = 0.995;% parametro penalidade do gradiente reduzido
        c = 0.0817; % parametro da busca
        dfdx = zeros(2*nb,1);
        dfdu = zeros(2*nb+na,1);
        dgdu = zeros(2*nb,2*nb+na);
        penal = zeros(nb,1);
        penal_d = zeros(nb,1);
        lambda = zeros(2*nb,1);
        %         [P,dP,Q,dQ,Pperdas,fo_a,dfdx,lambda,dfdu,dgdu,dldu,V,teta]=gradred11nov_1(teta,V,Pg,Pc,Qg,Qc,nb,nlin,na,nu,tipo,akm,gkm,bsh,bkm,bkmsh,o,d,c,Qmin,Qmax,Vmin,Vmax,Snapshot,G,B,w,penal,penal_d,PV,SW,Bus,Line,npv);
        pivo=0;
        soma = 0;
    end
    
    
    x_v = repmat(ones,npart,nb);
    for i=1:nb
        x_v(:,i) = V(i);
    end
    
    x_tap = repmat(ones,npart,na);
    for i=1:npart
        for z=1:na
            x_tap(i,z)=tap(1,z);
        end
    end
    
    shunt = zeros(1,nsh);
    for i = 1:size(Shunt.con,1)
        %         if Shunt.con(i,6)~=0
        shunt(:,i) = Shunt.con(i,6);
        %         end
    end
    x_sh = zeros(npart,nsh);
    for i=1:npart
        x_sh(i,:)=shunt(1,:) ;
    end
    
    %% vetor de particulas
    x = [x_v x_tap x_sh];
    
    tensao = zeros(npart,nb);
    taps = zeros(npart,na);
    shunts = zeros(npart,nsh);
    %=============================
    
    
    %% parametro inercia W para variavel TENSAO
    w_v_min = 0;
    w_v_max = 0.35;
    
    w_v = repmat(w_v_min,npart,3);
    wmin_v = repmat(w_v_min,npart,3);
    wmax_v = repmat(w_v_max,npart,3);
    
    w1_v = w_v_min + (w_v_max - w_v_min).*rand(npart,1);
    w2_v = w_v_min + (w_v_max - w_v_min).*rand(npart,1);
    w3_v = w_v_min + (w_v_max - w_v_min).*rand(npart,1);
    
    iw_v = [w1_v w2_v w3_v];
    
    
    %% parametro inercia W para variavel TAP
    w_t_min = 0;
    w_t_max = 0.2;
    
    w_t = repmat(w_t_min,npart,3);
    wmin_t = repmat(w_t_min,npart,3);
    wmax_t = repmat(w_t_max,npart,3);
    
    w1_t = w_t_min + (w_t_max - w_t_min).*rand(npart,1);
    w2_t = w_t_min + (w_t_max - w_t_min).*rand(npart,1);
    w3_t = w_t_min + (w_t_max - w_t_min).*rand(npart,1);
    
    iw_t = [w1_t w2_t w3_t];
    
    %% parametro inercia W para variavel SHUNT
    w_sh_min = 0;
    w_sh_max = 0.35;
    
    w_sh = repmat(w_sh_min,npart,3);
    wmin_sh = repmat(w_sh_min,npart,3);
    wmax_sh = repmat(w_sh_max,npart,3);
    
    w1_sh = w_sh_min + (w_sh_max - w_sh_min).*rand(npart,1);
    w2_sh = w_sh_min + (w_sh_max - w_sh_min).*rand(npart,1);
    w3_sh = w_sh_min + (w_sh_max - w_sh_min).*rand(npart,1);
    
    iw_sh = [w1_sh w2_sh w3_sh];
    
    
    iw = [iw_v iw_t iw_sh];
    
    % inicializacao do parametro vel (velocidade)
    %=============================
    % velocidade tensao
    vel_tensao_min = 0;
    vel_tensao_max = 0.01;
    vel_tensao = vel_tensao_min + (vel_tensao_max - vel_tensao_min).*rand(npart,nb);
    % velocidade tap
    vel_tap_min = 0;
    vel_tap_max = 0.001;
    vel_tap = vel_tap_min + (vel_tap_max - vel_tap_min).*rand(npart,na);
    % velocidade shunt
    vel_sh_min = 0;
    vel_sh_max = 0.1;
    vel_sh = vel_sh_min + (vel_sh_max - vel_sh_min).*rand(npart,nsh);
    
    if sistem==300
        % inicializacao do parametro vel (velocidade)
        %=============================
        % velocidade tensao
        vel_tensao_min = 0;
        vel_tensao_max = 0.1;
        vel_tensao = vel_tensao_min + (vel_tensao_max - vel_tensao_min).*rand(npart,nb);
        % velocidade tap
        vel_tap_min = 0;
        vel_tap_max = 0.01;
        vel_tap = vel_tap_min + (vel_tap_max - vel_tap_min).*rand(npart,na);
        % velocidade shunt
        vel_sh_min = 0;
        vel_sh_max = 0.01;
        vel_sh = vel_sh_min + (vel_sh_max - vel_sh_min).*rand(npart,nsh);
    end
    
    
    %=============================
    % parametro da distribuicao de gauss
    %=============================
    t = 0.25; % parametro de dispersao de aprendizagem do peso w. OBS: valores muito altos (20) as interacoes ficam mais demoradas e algumas variaveis ultrapassam seu limite (Switch SW bus <Bus  1> to theta-Q bus: Max Qg reached)
    t_= 0.02; % parametro de dispersao de ruido do gbest. OBS: valor muito alto, a funcao objetivo nao se altera
    
    % variavel para plotar tap
    conv_tap = zeros(npart,na);
    conv_shunt = zeros(npart,nsh);
    
    % parametros da funcao senoidal
    B = 1; % expoente da funcao senoidal
    L_0 = 10^-6; % gama = amplitude inicial
    L = repmat(L_0,npart,1); % amplitude da funcao senoidal
    c_sen = 10; % parametro que multiplica L
    E= 10^-3;
    
    % parametros da funcao polinomial
    B1 = 1;
    ni_0 = 10^-6;
    ni = repmat(ni_0,npart,1);
    c_poli = 10;
    E1 = 10^-3;
    
    % parametros penalidade
    alfa_v = 10000;
    alfa_q = 10000;

    %=============================
    
    %% CABEÇALHO
    [sistem,npart,itermax,t,t_,yy] = cabecalho (meta,modelo,sistem,npart,itermax,yy,t,t_);
    
    %% Inicio EPSO
    t3 = clock;
    timepbest = 0;
    timefc = 0;
    
    while iter < itermax
        tic
        %% Utilizando GRADIENTE
        if modelo==51 || modelo == 52 || modelo == 53 || modelo == 54
            if pivo == 1 % pivo = 1 usa gradiente
                
                cont_grad = cont_grad+1; % contador da quant. de iter. do gradiente
                
                if sistem==57
                    [P,dP,Q,dQ,Pperdas,fo_a,dfdx,lambda,dfdu,dgdu,dldu,V,teta]=gradred11nov_57b(teta,V,Pg,Pc,Qg,Qc,nb,nlin,na,nu,tipo,akm,gkm,bsh,bkm,bkmsh,o,d,c,Qmin,Qmax,Vmin,Vmax,Snapshot,G,B,w,penal,penal_d,PV,SW,Bus,Line,npv);
                end
                if sistem==118
%                     [P,dP,Q,dQ,Pperdas,fo_a,dfdx,lambda,dfdu,dgdu,dldu,V,teta]=gradred_57b(teta,V,Pg,Pc,Qg,Qc,nb,nlin,na,nu,tipo,akm,gkm,bsh,bkm,bkmsh,o,d,c,Qmin,Qmax,Vmin,Vmax,Snapshot,G,B,w,penal,penal_d,PV,SW,Bus,Line,npv);
                    [P,dP,Q,dQ,Pperdas,fo_a,dfdx,lambda,dfdu,dgdu,dldu,V,teta]=gradred_semtap(teta,V,Pg,Pc,Qg,Qc,nb,nlin,na,nu,tipo,akm,gkm,bsh,bkm,bkmsh,o,d,c,Qmin,Qmax,Vmin,Vmax,Snapshot,G,B,w,penal,penal_d,PV,SW,Bus,Line,npv);

                end
                if sistem == 300
%                     [P,dP,Q,dQ,Pperdas,fo_a,dfdx,lambda,dfdu,dgdu,dldu,V,teta]=gradred11nov_1_300b(teta,V,Pg,Pc,Qg,Qc,nb,nlin,na,nu,tipo,akm,gkm,bsh,bkm,bkmsh,o,d,c,Qmin,Qmax,Vmin,Vmax,Snapshot,G,B,w,penal,penal_d,PV,SW,Bus,Line,npv);
                    [P,dP,Q,dQ,Pperdas,fo_a,dfdx,lambda,dfdu,dgdu,dldu,V,teta]=gradred_semtap(teta,V,Pg,Pc,Qg,Qc,nb,nlin,na,nu,tipo,akm,gkm,bsh,bkm,bkmsh,o,d,c,Qmin,Qmax,Vmin,Vmax,Snapshot,G,B,w,penal,penal_d,PV,SW,Bus,Line,npv);
                end
                if sistem == 14 || sistem == 30 % || sistem ==118
                    [P,dP,Q,dQ,Pperdas,fo_a,dfdx,lambda,dfdu,dgdu,dldu,V,teta]=gradiente(teta,V,Pg,Pc,Qg,Qc,nb,nlin,na,nu,tipo,akm,gkm,bsh,bkm,bkmsh,o,d,c,Qmin,Qmax,Vmin,Vmax,Snapshot,G,B,w,penal,penal_d,PV,SW,Bus,Line,npv);
                end
                
                % substituindo as variaveis que saem do gradiente
                for i=1:nb
                    x_v(:,i) = V(i);
                end
                
                for i=1:npart
                    for z=1:na
                        x_tap(i,z)=tap(1,z);
                    end
                end
                
                for i = 1:size(Shunt.con,1)
                    shunt(:,i) = Shunt.con(i,6);
                end
                
                for i=1:npart
                    x_sh(i,:)=shunt(1,:) ;
                end
                
                % vetor de particulas
                x = [x_v x_tap x_sh];
                
                % ajustes nas variaveis de controle tensao para modelo
                for i=1:nb
                    for z=1:npart
                        if  x(z,i)<= Vmin
                            x(z,i)= Vmin;
                        end
                        if x(z,i)>= Vmax
                            x(z,i)= Vmax;
                        end
                    end
                end
            end
            pivo=0;
        end
        
        
        %% looping para criar PBEST
        %=============================
        t5=clock;
        for j=1:size(pbest,1)
            
            
            % Incluir as tensoes das barras
            %=============================
            % entrada: x(j,1:nu)  -> Vuo_
            Vuo_ = x(j,1:nb); % saida: Vuo
            Vu = zeros(nb,1);
            %             l = 1;
            for i=1:nb
                % if tipo(i)~=0 % colocar em Vu apenas as magnt de tensao das variaveis de controle
                Vu(i) = Vuo_(1,i); % colocando todas as variaveis de tensao
                % l = l + 1;
                % end
            end
            %=============================
            
            
            % Incluindo as tensoes das barras PV's
            %=============================
            PV_ = PV.store;
            for i=1:nb
                if tipo(i)==1
                    
                    % armazendo o numero da barra atual
                    barra_atual = Bus.con(i,1);
                    
                    % percorrendo as barras da estruta PV.store
                    jj=1;
                    for n=1:npv
                        if PV_(n,1) == barra_atual
                            jj =n;
                            break % encontrou a barra atual
                        end
                    end
                    
                    % atualizando a magnitude de tensao
                    PV.store(jj,5) = Vu(i);
                elseif tipo(i)==2 % para barra slack
                    SW.store(1,4) = Vu(i);
                end
            end
            %=============================
            
            
            %% PARA MODELOS 2 ou 3 ou 4
            if modelo ==2 || modelo==3 || modelo==4 || modelo == 52 || modelo == 53 || modelo == 54
                
                %Atualizar no psat os valores atuais dos TAPS da matriz x
                %entrada: x(j,nu+1:nu+na)  -> Line.store(:,11)
                %=============================
                jj = 1;
                for i=1:size(Line.store(:,11),1)
                    if(Line.store(i,11) ~= 0)
                        Line.store(i,11) = x(j,nb+jj) ;
                        jj =jj + 1;
                    end
                end
                
                %Atualizar no psat os valores atuais dos SHUNTs da matriz x
                jj = 1;
                for i=1:size(Shunt.store(:,6),1)
                    Shunt.store(i,6) = x(j,nb+na+jj) ;
                    jj =jj + 1;
                end
                
            end
            %% =============================
            
            
            % executando o Fluxo de Carga
            %=============================
            runpsat('pf');
            timefc = timefc + Settings.lftime;
            %=============================
            
            % atribuindo os valores do PSAT as vars magnitude de tensao e
            % angulos
            %=============================
            V = Snapshot.y(1+nb:2*nb);
            teta = Snapshot.y(1:nb);
            %=============================
            
            %% MODELO 2 ou 3 ou 4
            %% =============================
            if modelo==2 || modelo==3 || modelo==4 || modelo == 52 || modelo == 53 || modelo == 54
                
                % atribuindo os valores do PSAT as vars tap
                akm = Line.con(:,11);
                jj = 1;
                for i=1:nlin
                    if Line.con(i,11)~= 1.00
                        tap(jj)=Line.con(i,11);
                        jj = jj + 1;
                    end
                end
                
                % SHUNT
                shunt(:,1) = Shunt.con(1,6);
                ii=1;
                for i=1:nb
                    if bsh(i,1)~= 0
                        bsh(i,1)=Shunt.con(ii,6);
                        ii=ii+1;
                    end
                end
            end
            %=============================
            
            %% restricoes
            % equacoes potencias nodais
            % Equacao das barras de tipo 0 ou 1
            %--------------- Delta P ---------------%
            [P,dP] = calculap(teta,V,Pg,Pc,nb,nlin,tipo,akm,gkm,bkm,o,d);
            
            % Equacao das barras de tipo 0
            %--------------- Delta Q ---------------%
            [Q,dQ] = calculaq(teta,V,Qg,Qc,nb,nlin,tipo,akm,gkm,bkm,bsh,bkmsh,o,d);
            
            
            % Formulando a funcao perdas
            %=============================
            for i=1:nlin
                kk = o(i);
                m = d(i);
                Pperdas(i) = gkm(i)*( (1/akm(i)^2)*V(kk)^2 + V(m)^2 - 2*(1/akm(i)) * V(kk)*V(m)*cos(teta(kk) - teta(m)));
            end
            %=============================
            
            
            %% PENALIDADES
            %=============================
            [penal1,penal2,penal3]=penalidade(npart,nb,tipo,V,Vmin,Vmax,Snapshot,Qmin,Qmax);
            %=============================
            
            %% FUNCAO OBJETIVO MODELOS CONTINUOS
            if modelo==1 || modelo==2 || modelo==4 || modelo==51 || modelo==52 || modelo==54
                fo(j) = sum(Pperdas) + alfa_v * sum(penal1);
                fperdas(j)=sum(Pperdas);
            end
            %% FUNCAO OBJETIVO MODELO CONTINUO COM MARGEM DE REATIVOS
            if modelo == 61
%                 fo(j) = W*sum(Pperdas) + alfa_v * sum(penal1) + (1-W)*sum(penal3);
                fperdas(j) = sum(Pperdas);
                geracao(j) = sum(penal3);
                fo(j) = W*((sum(Pperdas) - sum_min_loss)/(sum_max_loss - sum_min_loss)) + (1-W)*((sum(penal3) - sum_min_reactive)/(sum_max_reactive-sum_min_reactive));
                
                
            end
            
            %% criando vetor de discretos para shunt
            [disc_sh] = discretos_shunt(sistem);
            
            
            %% FUNCAO OBJETIVO PARA MODELOS DISCRETOS
            if modelo == 3 || modelo == 53
                
                % FUNCAO POLINOMIAL PARA SHUNTS
                [psi]=polinomial_sh(disc_sh,shunt,nsh,B1);
                
                % FUNCAO SENOIDAL PARA TAPS
                fi = zeros(1,na);
                for i=1:na
                    fi(i) = sin((tap(i)/0.0075)*pi+2.09439510239332).^(2*B);
                end
                
                % FUNCAO OBJETIVO AUMENTADA
                fo(j) = sum(Pperdas) + alfa_v * sum(penal1)  + L(j) * sum(fi) + ni(j) * sum(psi);
                fperdas(j)=sum(Pperdas);
                
            end
            
%              %% FUNCAO OBJETIVO PARA MODELOS DISCRETOS COM MARGEM DE REATIVOS
%             if modelo == 63
%                 
%                 % FUNCAO POLINOMIAL PARA SHUNTS
%                 [psi]=polinomial_sh(disc_sh,shunt,nsh,B1);
%                 
%                 % FUNCAO SENOIDAL PARA TAPS
%                 fi = zeros(1,na);
%                 for i=1:na
%                     fi(i) = sin((tap(i)/0.0075)*pi+2.09439510239332).^(2*B);
%                 end
%                 
%                 % FUNCAO OBJETIVO AUMENTADA
%                 fo(j) = sum(Pperdas) + alfa_v * sum(penal1)  + L(j) * sum(fi) + ni(j) * sum(psi) + Wq * sum(penal3);
%                 fperdas(j)=sum(Pperdas);
%                 
%             end
            
            % atualizando as particulas
            %=============================
            tensao(j,1:nb) = V';
            taps(j,1:na) = tap;
            shunts(j,1:nsh) = shunt;
            x(j,1:nb) = tensao(j,1:nb);
            x(j,nb+1:nb+na) = taps(j,1:na);
            x(j,nb+na+1:nb+na+1) = shunts(j,1:1);
            %=============================
            
            % transpondo deltap e deltaq
            dp = dP';
            dq = dQ';
            Qg = Snapshot.Qg';
            
            % criando o PBEST
            %=============================
            Vpb(j,:) = x(j,1:nb);
            Apb(j,:) = x(j,nb+1:nb+na);
            Spb(j,:) = x(j,nb+na+1:nb+na+nsh);
            deltap(j,:) = dp(1,1:nb);
            deltaq(j,:) = dq(1,1:nb);
            fopb(j) = fo(j);
            Qgpb(j,:) = Qg(1,1:nb);
            
            % pbest
            pbest = [Vpb Apb fopb deltap deltaq fperdas Qgpb Spb];
            %=============================
            
        end % fim do calculo do pbest
        
        t6=clock;
        tpbest = etime(t6,t5); % calculo do tempo total do PBEST em segundos
        timepbest = timepbest + tpbest;
        
        
        %% olhando cada linha do vetor PBEST e pegando apenas o menor valor da funcao objetivo das
        % iteracoes
        if modelo==1 || modelo==51 || modelo==61
            if iter < 1
                % calculo do pbest
                pbest0 = pbest;
            end
        end
        if modelo==2 || modelo==3 || modelo==4 || modelo == 52 || modelo == 53 || modelo == 54
            % isso é feito pq com a mudanca dos taps e shunts, a primeira iteracao é sempre melhor
            % que a segunda e se isso acontecer nao há mudanca. (os taps e shunts da primeira iteracao ainda nao sofreu alteracao pra discretizar)
            if iter <=1
                pbest0 = pbest;
            end
        end
        
        for i=1:size(pbest,1)
            if (pbest0(i,nb+na+1)<=pbest(i,nb+na+1)) % posicao da FOA nb+na+1
                pbest_new(i,:) = pbest0(i,:);
            else
                pbest_new(i,:) = pbest(i,:);
            end
        end
        
        %% GBEST
        %=============================
        [M,I] = min(pbest_new(:,nb+na+1)); % pego o minimo do PBEST e toda sua linha
        gbest = pbest_new (I,:); % o melhor do melhor
        reativo_gera = geracao (I,:);
        gbestnew = gbest(1,nb+na+1+nb+nb+1);
        %=============================
        
        % atribuicao dos valores apenas para o calculo, s/ valor da funcao obj
        %=============================
        pb = pbest_new(:,1:nb+na);
        pb (:,nb+na+1:nb+na+nsh) = pbest_new(:,nb+na+1+nb+nb+1+nb+1:nb+na+1+nb+nb+1+nb+nsh); % incluindo a variavel shunt que está no final do vetor pbest
        gb = gbest(1,1:nb+na);
        gb(1,nb+na+1:nb+na+nsh) = gbest(1,nb+na+1+nb+nb+1+nb+1:nb+na+1+nb+nb+1+nb+nsh);% incluindo a variavel shunt que está no final do vetor gbest
        %=============================
        
        
        
        %% ATUALIZACAO DOS PARAMETROS PESOS E VELOCIDADE DO EPSO e DEEPSO
        [vel_tensao,vel_tap,vel_sh] = param_update(PP,sg,rr,meta,x,pb,nb,na,nsh,npart,pbest,gbest_gauss,gb,t_,t,w_v,wmax_v,wmin_v,iter,itermax,iw_v,w_t,wmax_t,wmin_t,iw_t,w_sh,wmax_sh,wmin_sh,iw_sh,vel_tensao,vel_tap,vel_sh);
        
        
        if iter==1
            gbest0 = gbest(1,nb+na+1+nb+nb+1);
        end
        
        if iter ~=1
            % Teste para uso do modelo 51 GRADIENTE
            if modelo == 51 || modelo == 52 || modelo == 53 || modelo == 54 %&& iter>2
                if abs(gbest0 - gbest(1,nb+na+1+nb+nb+1))<=10^-5
                    soma = soma+1;
                    if soma == 10
                        pivo=1; % pivo = 1 usa gradiente
                        soma = 0;
                    end
                else
                    pivo=0; % pivo =0 nao usa gradiente
                end
            end
            
            % controlando quantidade de iteracoes de gradiente
            if cont_grad >= cont_grad_max
                pivo = 0; % se chegou ao numero maximo de iteracoes do gradiente, nao usa gradiente
            end
        end
        
        
        %% RELATORIO DE SAIDA
        [gbest]=relatorio_saida(rel_total,gbest,iter,itermax,nlin,nb,na,nsh,tmin,tmax,apasso,Shunt,Vmin,Vmax,tipo,Qmin,Qmax,Line,modelo,disc_sh,reativo_gera);
        
        
        %% ATUALIZACAO DAS VARIAVEIS DE CONTROLE
        [x] = update(nb,na,nsh,x,npart,tipo,vel_tensao,vel_tap,vel_sh,modelo);
        
        
        %% arredondamento das variaveis TENSAO e TAP
        if modelo==4 || modelo == 54
            [x] = rouding(nsh,nb,tipo,npart,x,Vmax,Vmin,na,tmin,tmax,apasso,disc_sh);
        end
        
        
        %% ATUALIZACAO DO PARAMETROS DA SENOIDAL E POLINOMIAL
        if modelo ==3 || modelo == 53
            % SENOIDAL
            [L] = gama_update(npart,nb,na,apasso,tmin,tmax,x,E,L,L_0,c_sen);
            % POLINOMIAL
            [ni] = ni_update(npart,nb,na,nsh,disc_sh,x,ni,ni_0,c_poli,E1);
        end
        
        
        %% acrescenta em iter
        iter = iter +1;
        
        %% variaveis para plotar
        conv_tap(iter,:) = gbest(1,nb+1:nb+na); % variavel para plotar o tap
        conv_shunt(iter,:) =  gbest(1,nb+na+1+nb+nb+1+nb+1:nb+na+1+nb+nb+1+nb+nsh);
        
        
        %% atribuo o valor do novo PBEST ao antigo
        pbest0 = pbest_new;
        gbest0 = gbestnew;
        toc
        
    end
    
    t4 = clock;
    %     fobj(yy)= gbest(nb+na+1+nb+nb+1)*100;
    
    
end

timemeta = etime(t4,t3); % calculo do tempo total do EPSO em segundos

% xlswrite(arquivo,fobj);

    fprintf('\n***************************************************************');
    fprintf('\n***************************************************************');
    fprintf('\n Tempo de execução EPSO: %f s', timemeta);
    fprintf('\n Tempo de execução FC: %f s', timefc);
    if modelo == 51 || modelo == 52 || modelo == 53 || modelo == 54
        fprintf('\n Quant. iter. grad.: %d', cont_grad);
    end
    fprintf('\n***************************************************************');
    fprintf('\n***************************************************************');

diary off



