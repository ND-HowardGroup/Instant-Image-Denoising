close all
clear variables
clc

%Varun Mannam
%date: 23rd Sep 2019
%calculate the PSNR on test-mix data
format long
font = 14;
size = 60;
lr02=[31.0013994757380,37.8707040566466;32.4355944495239,35.6309636442510;30.8389376260123,37.6492493218672;30.1446963613668,36.6236157602191;22.8898419300006,30.7631725906395;25.4380806137492,25.1707394965457;31.4129562671116,36.8197819466616;33.2714634385635,36.3603092914364;33.2687423523874,34.7392999015588;22.8804735366571,30.7644462985140;25.6158044368661,26.6756731401274;22.8279560741158,30.7384143418196;25.6370939543658,25.9890214809789;33.0234393128250,36.1915866501309;25.6158044368661,26.6756731401274;22.8580037151022,30.7422372168434;23.8972761467197,28.6079129275166;24.9514139990449,33.1290783190592;24.9832541276875,33.4505545189863;23.8755250603244,28.5235817855360;24.3188985682586,30.7802329084656;23.8605156142818,28.5273858572177;25.2263003138287,33.9323657702808;23.9044985921393,28.6042980049049;23.6688313676180,27.8257465031556;32.2013251950805,37.5028272902574;24.1923646677596,28.8594757041154;26.2300198751109,30.9162420443244;24.7010303920376,31.6063347609391;26.2356203085374,30.9508304564409;32.0904729461831,37.4220367419407;31.8870667992703,35.9612789041353;24.4086429479671,30.5335382967549;26.2348882344496,30.8970025229033;31.7707628427737,36.6106018163996;26.2283210436201,30.9095972926809;24.9907764180876,32.2287805187400;29.3709502307532,36.8771879131467;28.7487655843002,32.9158452702596;28.6149527091447,32.3612022308291;28.1247666427617,30.8798192112676;25.5706915349250,31.5466131561593;29.3276646439983,36.7703894344958;28.2432957509904,32.5893360072649;29.3725190518817,36.8387129137902;25.4384495835989,32.6905496685404;25.2179004485139,32.5280217288885;29.3606855385532,36.8420411060489];
lr01=[31.0013994757380,37.9720500569364;32.4355944495239,36.5569349573362;30.8389376260123,37.9228664143875;30.1446963613668,37.1864480945905;22.8898419300006,31.1312976907205;25.4380806137492,26.0806791928658;31.4129562671116,36.3967684087515;33.2714634385635,37.5246738075772;33.2687423523874,35.2824308627102;22.8804735366571,31.1598501207368;25.6158044368661,27.2175215987301;22.8279560741158,31.1997641633473;25.6370939543658,26.7208539584304;33.0234393128250,37.2677034274512;25.6158044368661,27.2175215987301;22.8580037151022,31.1910007081443;23.8972761467197,29.1169254960878;24.9514139990449,32.6325559053955;24.9832541276875,33.3448851462099;23.8755250603244,29.0648906629550;24.3188985682586,30.2393085802698;23.8605156142818,29.0013075200580;25.2263003138287,33.7494334175423;23.9044985921393,29.1544660087896;23.6688313676180,27.9404715126978;32.2013251950805,38.5162895836068;24.1923646677596,29.0121363277007;26.2300198751109,31.8550827041069;24.7010303920376,31.9482440778708;26.2356203085374,31.8939498518698;32.0904729461831,38.4166839362487;31.8870667992703,36.6141089708192;24.4086429479671,30.9582733788868;26.2348882344496,31.8710709829973;31.7707628427737,37.3798269549075;26.2283210436201,31.8660353089117;24.9907764180876,32.3493243316979;29.3709502307532,36.8441028117600;28.7487655843002,33.2583621758031;28.6149527091447,32.6456097551943;28.1247666427617,31.0726016201915;25.5706915349250,31.5030877157482;29.3276646439983,36.6502948875097;28.2432957509904,32.9361005165339;29.3725190518817,36.8150180181501;25.4384495835989,32.7805338526161;25.2179004485139,32.5920391480519;29.3606855385532,36.7781162637473];
lr1=[31.0013994757380,38.0450971838621;32.4355944495239,36.1718638560191;30.8389376260123,37.9540752776785;30.1446963613668,37.1042739251729;22.8898419300006,31.2966528519693;25.4380806137492,30.3806127324698;31.4129562671116,36.5421420141026;33.2714634385635,37.1829372314317;33.2687423523874,35.4163094071092;22.8804735366571,31.3018426684524;25.6158044368661,30.5441769755517;22.8279560741158,31.3330816618869;25.6370939543658,30.6681985848303;33.0234393128250,36.9170690371075;25.6158044368661,30.5441769755517;22.8580037151022,31.3114941347816;23.8972761467197,29.6080719707910;24.9514139990449,33.3533448189249;24.9832541276875,33.2641938865110;23.8755250603244,29.4557386920728;24.3188985682586,31.8281755886988;23.8605156142818,29.5212887117986;25.2263003138287,33.2749221647761;23.9044985921393,29.5463951969645;23.6688313676180,30.2874884403492;32.2013251950805,38.6235728660093;24.1923646677596,28.5709551857455;26.2300198751109,32.4955833061121;24.7010303920376,31.5341581313279;26.2356203085374,32.5452622058156;32.0904729461831,38.4729196277372;31.8870667992703,36.4689769122771;24.4086429479671,32.4028471667861;26.2348882344496,32.4602129291253;31.7707628427737,37.5698319023003;26.2283210436201,32.5208064872131;24.9907764180876,31.9468928267456;29.3709502307532,37.0503161622957;28.7487655843002,33.2315874864787;28.6149527091447,32.6745841524984;28.1247666427617,31.1754018268960;25.5706915349250,31.6301087503943;29.3276646439983,36.8778992462304;28.2432957509904,32.7833707874371;29.3725190518817,36.9912668615607;25.4384495835989,32.6873076846418;25.2179004485139,32.2697687655437;29.3606855385532,36.9590359603794];
lr2=[31.0013994757380,38.3101495278474;32.4355944495239,36.5580143695862;30.8389376260123,38.2290048370148;30.1446963613668,37.2018106464557;22.8898419300006,31.1588745377311;25.4380806137492,32.0974360555022;31.4129562671116,36.6608295404559;33.2714634385635,37.6288875700694;33.2687423523874,35.7597453872692;22.8804735366571,31.1945227531623;25.6158044368661,32.4922402715761;22.8279560741158,31.2321155204350;25.6370939543658,32.5893244451558;33.0234393128250,37.3444630910685;25.6158044368661,32.4922402715761;22.8580037151022,31.2283062729600;23.8972761467197,31.5677415908786;24.9514139990449,33.7925308620272;24.9832541276875,33.9681143106083;23.8755250603244,31.4284594910000;24.3188985682586,31.9450806480444;23.8605156142818,31.4184653884590;25.2263003138287,33.9765344093026;23.9044985921393,31.5557978359201;23.6688313676180,30.1205045161897;32.2013251950805,38.6425261922597;24.1923646677596,29.2048292174256;26.2300198751109,34.1850525992299;24.7010303920376,32.3977201655303;26.2356203085374,34.3598618050617;32.0904729461831,38.5259392750998;31.8870667992703,36.5878224103482;24.4086429479671,32.1312512419017;26.2348882344496,34.2329409867991;31.7707628427737,37.5536892197757;26.2283210436201,34.3534624385590;24.9907764180876,32.3726806239636;29.3709502307532,37.1797738904243;28.7487655843002,33.4592916954720;28.6149527091447,32.9253799045674;28.1247666427617,31.3941446635880;25.5706915349250,31.6690501289592;29.3276646439983,36.9881878352174;28.2432957509904,32.9576340458749;29.3725190518817,37.1185054876709;25.4384495835989,32.9273576343424;25.2179004485139,32.6966872430471;29.3606855385532,37.0626781907985];
lr3=[31.0013994757380,38.2239460456255;32.4355944495239,36.9002793535786;30.8389376260123,38.2166113639627;30.1446963613668,37.4059130191456;22.8898419300006,31.7875446507455;25.4380806137492,32.0369191998330;31.4129562671116,36.1759463795475;33.2714634385635,38.0530946126104;33.2687423523874,36.0724195138785;22.8804735366571,31.8119470181988;25.6158044368661,31.8880910259557;22.8279560741158,31.9414489961851;25.6370939543658,32.2992515645357;33.0234393128250,37.7752854777759;25.6158044368661,31.8880910259557;22.8580037151022,31.8253768807799;23.8972761467197,32.2356255137506;24.9514139990449,33.9154519257267;24.9832541276875,33.8182098147688;23.8755250603244,31.9673425327874;24.3188985682586,32.2228868565630;23.8605156142818,32.0699659052054;25.2263003138287,33.7420256445836;23.9044985921393,32.0859210877233;23.6688313676180,30.3693062126005;32.2013251950805,38.5736496671072;24.1923646677596,29.0719525414162;26.2300198751109,34.4462205001779;24.7010303920376,32.2970912516436;26.2356203085374,34.6905387005706;32.0904729461831,38.4497562565495;31.8870667992703,36.4456372116735;24.4086429479671,32.4932466663890;26.2348882344496,34.5063598042791;31.7707628427737,37.5613025776674;26.2283210436201,34.6536727656353;24.9907764180876,32.4914583846453;29.3709502307532,37.2440148373664;28.7487655843002,33.5071132765551;28.6149527091447,32.9036090355598;28.1247666427617,31.2785332456663;25.5706915349250,31.4932130745720;29.3276646439983,37.0515254564590;28.2432957509904,33.1861778261284;29.3725190518817,37.1401285817825;25.4384495835989,32.8599612798510;25.2179004485139,32.7381839690873;29.3606855385532,37.1058267403902];
lr4=[31.0013994757380,37.9022022209522;32.4355944495239,36.0354773655393;30.8389376260123,37.6897570561963;30.1446963613668,37.1232702337459;22.8898419300006,31.2195586410569;25.4380806137492,30.3648316815561;31.4129562671116,36.1922110592087;33.2714634385635,36.9065560083615;33.2687423523874,35.3119990113358;22.8804735366571,31.2677662965883;25.6158044368661,30.6941068506303;22.8279560741158,31.2428452170821;25.6370939543658,30.7192258024659;33.0234393128250,36.7216266077294;25.6158044368661,30.6941068506303;22.8580037151022,31.2736526694975;23.8972761467197,30.8313519337580;24.9514139990449,33.5925320981093;24.9832541276875,33.1645612817427;23.8755250603244,30.4516788567473;24.3188985682586,32.3072763991487;23.8605156142818,30.7461360680451;25.2263003138287,33.0959295494610;23.9044985921393,30.5196828804004;23.6688313676180,30.0485179582555;32.2013251950805,37.7759220687261;24.1923646677596,28.5402496873052;26.2300198751109,33.5606854372993;24.7010303920376,31.5239817882144;26.2356203085374,33.7320695503778;32.0904729461831,37.6972216090155;31.8870667992703,35.7166075049498;24.4086429479671,32.4012059917732;26.2348882344496,33.5079103336207;31.7707628427737,36.9410378869043;26.2283210436201,33.6747328361187;24.9907764180876,32.5017671654973;29.3709502307532,36.3621324381419;28.7487655843002,32.7879598299863;28.6149527091447,32.2471131498917;28.1247666427617,30.8073977216116;25.5706915349250,31.1137957012149;29.3276646439983,36.0802338577404;28.2432957509904,32.7315014852439;29.3725190518817,36.2926159962628;25.4384495835989,32.5320094675906;25.2179004485139,32.6526187945414;29.3606855385532,36.2401044245416];
lr5=[31.0013994757380,37.9557346308498;32.4355944495239,35.6018111128794;30.8389376260123,37.8509228013963;30.1446963613668,37.0528074692118;22.8898419300006,30.8227718292946;25.4380806137492,29.7772200655209;31.4129562671116,36.3819962752428;33.2714634385635,36.2649039073686;33.2687423523874,34.8045192264707;22.8804735366571,30.8992624436239;25.6158044368661,29.6180887540089;22.8279560741158,30.9356237462065;25.6370939543658,29.7931986086617;33.0234393128250,36.1259053505363;25.6158044368661,29.6180887540089;22.8580037151022,30.8985113038668;23.8972761467197,29.9023318068359;24.9514139990449,32.5559295723155;24.9832541276875,32.6462043434889;23.8755250603244,29.6610067382160;24.3188985682586,31.1348378018182;23.8605156142818,29.8170437895608;25.2263003138287,32.6368227194780;23.9044985921393,29.7424149858502;23.6688313676180,29.6732969221073;32.2013251950805,37.7391819398027;24.1923646677596,28.7613914662000;26.2300198751109,31.9723382330057;24.7010303920376,31.6130101065458;26.2356203085374,32.1287718877584;32.0904729461831,37.5998112847147;31.8870667992703,36.0537664144018;24.4086429479671,31.6176489518648;26.2348882344496,31.9674323068313;31.7707628427737,36.8038004117098;26.2283210436201,32.0498293083271;24.9907764180876,32.2743239985477;29.3709502307532,36.1961960338582;28.7487655843002,32.1778158374237;28.6149527091447,31.6819493224782;28.1247666427617,30.3887053653959;25.5706915349250,31.0521649561203;29.3276646439983,36.0122546425318;28.2432957509904,32.1243180990108;29.3725190518817,36.1266248156799;25.4384495835989,32.3351641999625;25.2179004485139,32.4045368295401;29.3606855385532,36.1105191910644];

op_snr1 = lr1(:,2);
op_snr2 = lr2(:,2);
op_snr3 = lr3(:,2);
op_snr4 = lr4(:,2);
op_snr5 = lr5(:,2);
op_snr6 = lr01(:,2);
op_snr7 = lr02(:,2);

x=[1:48]';
figure(1), 
plot(x,op_snr1,'b-*','Linewidth',1);
hold on
plot(x,op_snr2,'r-*','Linewidth',1);
plot(x,op_snr3,'g-*','Linewidth',1);
plot(x,op_snr4,'m-*','Linewidth',1);
plot(x,op_snr5,'c-*','Linewidth',1);
plot(x,op_snr6,'y-*','Linewidth',1);
plot(x,op_snr7,'k-*','Linewidth',1);
title('BN diff LR values PSNR plot - raw data')
legend('lr1=1e-3','lr2=5e-4','lr3=1e-4','lr4=5e-5','lr5=1e-5','lr01=5e-3','lr02=1e-2','Location','best');
xlabel('Test index');
ylabel('Test PSNR');
set(gca,'FontSize',font)


ip_snr = lr1(:,1);
figure(2);
scatter(ip_snr,op_snr1,size,'b');
hold on
scatter(ip_snr,op_snr2,size,'r');
scatter(ip_snr,op_snr3,size,'g');
scatter(ip_snr,op_snr4,size,'m');
scatter(ip_snr,op_snr5,size,'c');
scatter(ip_snr,op_snr6,size,'y');
scatter(ip_snr,op_snr7,size,'k');
xlabel('input SNR');
ylabel('output SNR');
title('Scatter psnr (input and estimated) - BN raw data')
legend('lr1=1e-3','lr2=5e-4','lr3=1e-4','lr4=5e-5','lr5=1e-5','lr01=5e-3','lr02=1e-2','Location','best');
set(gca,'FontSize',font)
