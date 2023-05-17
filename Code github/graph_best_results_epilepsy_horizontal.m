%% graph to plot all the results, statistical validation for Epilepsy Paper
% Mauro Pinto 21/07/2020

% patient per SOP

%% Data
filename = 'Results\All_Final_results.xlsx';
A = xlsread(filename,'SS');
B = xlsread(filename,'Surrogate');
%%
matrix = [A(:,2) A(:,5) A(:,8) A(:,11); 0.64 0.75 0.64 0.69];
surrogate = [B(:,1) B(:,4) B(:,7) B(:,10); 0.64 0.75 0.64 0.69];
%%
close all


matrix=[linspace(0,0,38); matrix'];
surrogate=[linspace(0,0,38); surrogate'];

my_database_patient_list=['402','8902','11002','16202','21902','23902','26102','30802','32702','45402',...
    '46702','50802','53402','55202','56402','58602','59102','60002','64702','75202',...
    '80702','85202','93402','93902','94402','95202','96002','98102','101702','102202',...
    '104602', '109502','112802','113902','114702','114902','123902',"Overall"];


h=figure("Position",[10, 10, 2000, 200])
imagesc(matrix)
ax = gca;
xax = ax.XAxis; % xax = get(ax,'XAxis'); 
set(xax,'TickDirection','out')
xax = ax.YAxis; % xax = get(ax,'XAxis'); 
set(xax,'TickDirection','out')

set(gca, 'XTick', 1:1:38, 'XTickLabel', my_database_patient_list);
set(gca, 'YTick', 1:1:5, 'YTickLabel', ["\bfStrat.","Control","BLW","SbR","DWE"]);
xtickangle(30)
xlabel('Patients')
% ylabel([{'Mininum Pre-Ictal'},{'Period (minutes)'}])
colormap(flipud(bone))
colorbar()
caxis([0,1])

[row, column] = find(surrogate == 1);
hold on
plot(column,row,"kd", 'MarkerFaceColor','w','DisplayName','Above Chance Level')
hold on

stratification_seizures_classification=[1,3,4,5,6,7,8,9,11,12,14,16,17,18];
stratification_seizures_sleep=[2,5,6,10,12,13,14,15,18];
stratification_seizures_circadian=[2,3,5,6,7,8,9,11,12,14,15,16];
stratification_seizures_activity=[1,2,3,4,6,9,11,12,14,15,17,18];
% legend('Above Chance Level','Seizures Classification','Sleep Stage','Circadian Cycle','Seizure Activity','Location','bestoutside')

plot(stratification_seizures_classification-0.25,linspace(1-0.25,1-0.25,length(stratification_seizures_classification)),"k*", 'MarkerFaceColor','w','MarkerSize',5,...
    'DisplayName','Seizures Classification')
hold on
plot(stratification_seizures_sleep-0.25,linspace(1+0.25,1+0.25,length(stratification_seizures_sleep)),"k+",'MarkerFaceColor','w','MarkerSize',5,...
     'DisplayName','Sleep Stage')
hold on
plot(stratification_seizures_circadian+0.25,linspace(1-0.25,1-0.25,length(stratification_seizures_circadian)),"k.",'MarkerFaceColor','w','MarkerSize',5,...
     'DisplayName','Circadian Cycle')
hold on
plot(stratification_seizures_activity+0.25,linspace(1+0.25,1+0.25,length(stratification_seizures_activity)),"kx",'MarkerFaceColor','w','MarkerSize',5,...
     'DisplayName','Seizure Activity')

legend('show','Location','bestoutside')

text(38,5,'\bf84%','FontSize',7,'HorizontalAlignment','center','Color','white')
text(38,4,'\bf86%','FontSize',7,'HorizontalAlignment','center','Color','white')
text(38,3,'\bf89%','FontSize',7,'HorizontalAlignment','center','Color','white')
text(38,2,'\bf84%','FontSize',7,'HorizontalAlignment','center','Color','white')

for i=1:37
    a=line([i+0.5 i+0.5],[0 6],'LineStyle','--','Color','k','HandleVisibility','off');
    a.Color=[0,0,0,0.5];
end

a=line([38.5 38.5],[5 5],'LineStyle','--','Color','k','HandleVisibility','off');
a.Color=[0,0,0,0.9];
a=line([0 38.5],[4.5 4.5],'LineStyle','--','Color','k','HandleVisibility','off');
a.Color=[0,0,0,0.9];
a=line([0 38.5],[3.5 3.5],'LineStyle','--','Color','k','HandleVisibility','off');
a.Color=[0,0,0,0.3];
a=line([0 38.5],[2.5 2.5],'LineStyle','--','Color','k','HandleVisibility','off');
a.Color=[0,0,0,0.3];
a=line([0 38.5],[1.5 1.5],'LineStyle','--','Color','k','HandleVisibility','off');
a.Color=[0,0,0,0.9];
hold off

set(h,'PaperOrientation','landscape');
print('resultados2','-dpdf','-bestfit')


