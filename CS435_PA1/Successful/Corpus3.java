import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

public class Corpus3 {

    public static final int NUMBER_OF_REDUCERS_J1 = 8;
    public static final int FINAL_NO_OF_UNIGRAMS = 500;
    public static final String OUTPUT_PATH_FOR_JOB1 = "./intermediate";
    public static final String SEPERATOR = "\t";

    private static Logger LOGGER = Logger.getLogger(Corpus3.class.getName());

    public static class Mapper1 extends Mapper<Object, Text, Text, IntWritable> {

        private TreeMap<Text, Integer> tmap = null;
        private Text word = new Text();
        private IntWritable one = new IntWritable(1);

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            //preprocessing of input
            if (!value.toString().isEmpty()) {
                StringTokenizer itr = new StringTokenizer(value.toString().split("<====>")[2]);

                while (itr.hasMoreTokens()) {

                    String out = itr.nextToken().replaceAll("[^A-Za-z0-9]", "").toLowerCase();
                    word.set(out);

                    context.write( word, one );
                }
            }
        }
    }

    public static class Reducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {

            int count = 0;

            for (IntWritable intWritable : values) {
                count += 1;
            }

            context.write( key, new IntWritable(count) );
        }
    }

    public static class UnigramPartitioner extends Partitioner<Text, IntWritable> {
        @Override
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {

            if( key.toString().isEmpty() )
            {
                return  0;
            }
            else
            {
                Character partitionKey = key.toString().toLowerCase().charAt(0);

                if (partitionKey >= 'a' && partitionKey < 'e') {
                    return 1;
                } else if (partitionKey >= 'e' && partitionKey < 'i') {
                    return 2;
                } else if (partitionKey >= 'i' && partitionKey < 'm') {
                    return 3;
                } else if (partitionKey >= 'm' && partitionKey < 'p') {
                    return 4;
                } else if (partitionKey >= 'p' && partitionKey < 't') {
                    return 5;
                } else if (partitionKey >= 't' && partitionKey < 'w') {
                    return 6;
                } else if (partitionKey >= 'w' && partitionKey < 'z') {
                    return 7;
                } else {
                    return 0;
                }
            }
        }
    }

    public static class Mapper2 extends Mapper<Object, Text, Text, NullWritable> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

//            LOGGER.info("****R2****" + value.toString() );
            // {key, value} -> {(unigram\tfrequence),null}
            context.write( value, NullWritable.get() );

        }
    }

    public static class Reducer2 extends Reducer<Text, NullWritable, Text, NullWritable> {

        private static int count = 0;
        private static List<Text> sortedList = new ArrayList<Text>();


        @Override
        protected void reduce(Text key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {

//            LOGGER.info( "=========" + key.toString() + "========");
//            LOGGER.info("++++++++++" + count + "++++++++++++");
            if(count < FINAL_NO_OF_UNIGRAMS)
            {
                sortedList.add(new Text( key ));
                count += 1;
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {

//            LOGGER.info("---------------------SIZE of LIST is : " + sortedList.size() + "-------------------");
            for( Text unigram : sortedList )
            {
                context.write( unigram, NullWritable.get() );
            }
        }
    }

    public static class KeyComparator extends WritableComparator{

        protected KeyComparator()
        {
            super(Text.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            Text t1 = (Text) a;
            Text t2 = (Text) b;

            String[] t1Items = t1.toString().split(SEPERATOR);
            String[] t2Items = t2.toString().split(SEPERATOR);

            Integer frequency1 = Integer.parseInt(t1Items[1]);
            Integer frequency2 = Integer.parseInt(t2Items[1]);

            Integer comparison = frequency1.compareTo(frequency2);

            if (comparison == 0)
                return 1;
            else
                return -1 * comparison;

        }
    }


        public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job1 = Job.getInstance(conf, "Job 1");
        job1.setJarByClass(Corpus3.class);
        job1.setMapperClass(Mapper1.class);
        job1.setPartitionerClass(UnigramPartitioner.class);
        job1.setReducerClass(Reducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        job1.setNumReduceTasks(NUMBER_OF_REDUCERS_J1);
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(OUTPUT_PATH_FOR_JOB1));
        job1.waitForCompletion(true);

        Job job2 = Job.getInstance(conf, "Job 2");
        job2.setJarByClass(Corpus3.class);
        job2.setMapperClass(Mapper2.class);
        job2.setReducerClass(Reducer2.class);
        job2.setSortComparatorClass(KeyComparator.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(NullWritable.class);
        job2.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job2, new Path(OUTPUT_PATH_FOR_JOB1));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));
        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }


}
