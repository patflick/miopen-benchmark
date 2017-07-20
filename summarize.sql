.header on
.mode tabs
.import resnet50.tsv resnet50

create table cleaned as select * from resnet50 where not Module like "Sequential%" and not Module like "Short%";

create table summary as select Module, Dir, sum(Time) as SumTime, avg(Time) as
    AvgTime from cleaned group by Module, Dir order by Time desc;

.output resnet50_summary.tsv
select * from summary order by SumTime desc;
.quit
