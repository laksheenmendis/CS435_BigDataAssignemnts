import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

public class ProfileA{

    public static final int NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 = 10;
    public static final int NUMBER_OF_REDUCERS_FOR_JOB_3 = 13;

    public static String SEPERATOR = "\t";
    public static final String OUTPUT_PATH_FOR_JOB1 = "./intermediateA1";
    public static final String OUTPUT_PATH_FOR_JOB2 = "./intermediateA2";

    private static Logger LOGGER = Logger.getLogger(ProfileA.class.getName());

    public static class CountersClass {
        public static enum N_COUNTERS {
            DOCCOUNT
        }
    }

    public static class Mapper1 extends Mapper<Object, Text, Text, Text> {

        private Text word = new Text();
        private final static Text one = new Text("1");

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            //preprocessing of input
            if (!value.toString().isEmpty()) {
                String[] splitArr = value.toString().split("<====>");

                //check whether the body is there for a particular article
                if (splitArr.length == 3) {
                    String documentID = splitArr[1];
                    StringTokenizer itr = new StringTokenizer(splitArr[2]);

                    while (itr.hasMoreTokens()) {

                        String unigram = itr.nextToken().replaceAll("[^A-Za-z0-9]", "").toLowerCase();
                        if (!unigram.isEmpty()) {
                            if (!unigram.isEmpty()) {
                                String outMapperKey = documentID + SEPERATOR + unigram;
                                word.set(outMapperKey);
                                //key->DocumentID\tunigram, value->one
                                context.write(word, one);
                            }
                        }
                    }
                }
            }
        }
    }

    public static class DocUnigramPartitioner extends Partitioner<Text, Text> {

        @Override
        public int getPartition(Text key, Text value, int numReduceTasks) {

            String [] arr = key.toString().split(SEPERATOR);
            int docId = Integer.valueOf(arr[0]);

            //check document ID and decide the partition/reducer
            if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 0) {
                return 0;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 1) {
                return 1;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 2) {
                return 2;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 3) {
                return 3;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 4) {
                return 4;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 5) {
                return 5;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 6) {
                return 6;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 7) {
                return 7;
            } else if (docId % NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 == 8) {
                return 8;
            } else {
                return 9;
            }
        }
    }

    public static class Reducer1 extends Reducer< Text , Text, Text, Text> {

        /*
         * The input {key, value} for Reducer1 is {(DocumentID\tunigram),[one, one, one, ....]}
         * The output {key, value} of the Reducer1 is {DocumentID,(unigram/tfrequency)}
         * */
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            int sum = 0;
            for (Text val : values) {
                sum += 1;
            }

            //process the key->DocumentID\tunigram and separate DocumentID and Unigram
            String[] inputKeyArr = key.toString().split(SEPERATOR);

            if (inputKeyArr.length == 2) {
                //output {key, value} -> {DocumentID, (unigram\tfrequency)}
                Text docId = new Text(inputKeyArr[0]);
                Text unigramFreq = new Text(inputKeyArr[1] + SEPERATOR + String.valueOf(sum));

                context.write(docId, unigramFreq);
            }
        }
    }

    public static class Mapper2 extends Mapper<LongWritable, Text, Text, Text>{

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //here value would be in the form {DocumentID\tunigram\tfrequency}
            //hence, we need to do some preprocessing
            String[] arr = value.toString().split(SEPERATOR);

            if (arr.length == 3) {
                //{key, value} -> {DocumentID, (unigram\tfrequency)}
                context.write(new Text(arr[0]), new Text(arr[1] + SEPERATOR + arr[2]));
            }
        }
    }

    public static class Reducer2 extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            //increment the Counter, to keep track of the total number of documents in the entire corpus
            context.getCounter(CountersClass.N_COUNTERS.DOCCOUNT).increment(1);

            List<Text> unigramFrqList = new ArrayList<Text>();

            int maxFreq = 0;
            //{key,value}->{DocumentId, [list of (unigram\tfrequency)]}
            //find out the maximum frequency
            for (Text val : values) {
                unigramFrqList.add(new Text(val));
                String[] arr = val.toString().split(SEPERATOR);
                int freq = Integer.valueOf(arr[1]);
                if (freq > maxFreq) {
                    maxFreq = freq;
                }
            }

            //for each unigram, calculate TF value for each unigram
            for (Text val : unigramFrqList) {
                String[] arr = val.toString().split(SEPERATOR);

                double augmentedTF = 0.5 + 0.5 * ((double)Integer.parseInt(arr[1])/(double)maxFreq);
                LOGGER.info( val + " TF value is " + String.valueOf(augmentedTF) );
                //{key,value}->{DocumentId, (unigram\tTFvalue)}
                context.write( new Text(key), new Text( arr[0] + SEPERATOR + String.valueOf(augmentedTF) ) );
            }
        }
    }

    public static class Mapper3 extends Mapper<LongWritable, Text, Text, Text>{

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //here value would be in the form {DocumentID\tunigram\tTFvalue}
            //hence, we need to do some processing
            String[] arr = value.toString().split(SEPERATOR);

            LOGGER.info( "Key is :" + key + " Value is :" + value );
            if (arr.length == 3) {

                LOGGER.info( "Output Key is :" + arr[1] + " Output Value is :" + value );
                //{key, value} -> {unigram, (DocumentID\tunigram\tTFvalue)}
                context.write(new Text(arr[1]), new Text(value));
            }
        }
    }

    public static class UnigramPartitioner extends Partitioner<Text, Text> {
        @Override
        public int getPartition(Text key, Text value, int numReduceTasks) {

            Character partitionKey = key.toString().toLowerCase().charAt(0);

            if (partitionKey >= 'a' && partitionKey < 'c') {
                return 1;
            } else if (partitionKey >= 'c' && partitionKey < 'e') {
                return 2;
            } else if (partitionKey >= 'e' && partitionKey < 'g') {
                return 3;
            } else if (partitionKey >= 'g' && partitionKey < 'i') {
                return 4;
            } else if (partitionKey >= 'i' && partitionKey < 'k') {
                return 5;
            } else if (partitionKey >= 'k' && partitionKey < 'm') {
                return 6;
            } else if (partitionKey >= 'm' && partitionKey < 'o') {
                return 7;
            } else if (partitionKey >= 'o' && partitionKey < 'q') {
                return 8;
            } else if (partitionKey >= 'q' && partitionKey < 's') {
                return 9;
            } else if (partitionKey >= 's' && partitionKey < 'u') {
                return 10;
            } else if (partitionKey >= 'u' && partitionKey < 'w') {
                return 11;
            } else if (partitionKey >= 'w' && partitionKey < 'y') {
                return 12;
            } else {
                return 0;
            }
        }
    }

    public static class Reducer3 extends Reducer<Text, Text, Text, Text> {

        private int docCount;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            this.docCount  = context.getConfiguration().getInt(CountersClass.N_COUNTERS.DOCCOUNT.name(), 0);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            int unigramDocCount=0;
            List<Text> unigramList = new ArrayList<Text>();

            //{key, value} -> {unigram, [list of (DocumentID\tunigram\tTFvalue)]}
            for(Text text: values)
            {
                unigramDocCount += 1;
                unigramList.add(new Text(text));
            }

            //at the end of the for loop, we have found the number of documents in which a particular unigram occured
            double inverseDocFreq = Math.log10((double)docCount/unigramDocCount);

            for (Text concatText : unigramList)
            {
                String [] arr = concatText.toString().split(SEPERATOR);

                if( arr.length == 3 )
                {
                    double tfidf = Double.parseDouble( arr[2] ) * inverseDocFreq;
                    context.write( new Text(arr[0]), new Text(key + SEPERATOR+ String.valueOf(tfidf) ) );
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job1 = Job.getInstance(conf, "Job 1");
        job1.setJarByClass(ProfileA.class);
        job1.setMapperClass(Mapper1.class);
        job1.setReducerClass(Reducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        //set multiple reducers
        if( !args[2].equals("1") )
        {
            job1.setPartitionerClass(DocUnigramPartitioner.class);
            job1.setNumReduceTasks(NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2);
        }
        else
        {
            job1.setNumReduceTasks(1);
        }

        job1.setNumReduceTasks(NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2);
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(OUTPUT_PATH_FOR_JOB1));
        job1.waitForCompletion(true);

        Job job2 = Job.getInstance(conf, "Job 2");
        job2.setJarByClass(ProfileA.class);
        job2.setMapperClass(Mapper2.class);

        if( !args[2].equals("1") )
        {
            job2.setPartitionerClass(DocUnigramPartitioner.class);
            job2.setNumReduceTasks(NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2);
        }
        else
        {
            job2.setNumReduceTasks(1);
        }

        job2.setReducerClass(Reducer2.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job2, new Path(OUTPUT_PATH_FOR_JOB1));
        FileOutputFormat.setOutputPath(job2, new Path(OUTPUT_PATH_FOR_JOB2));
        job2.waitForCompletion(true);

        Counter documentCount = job2.getCounters().findCounter(CountersClass.N_COUNTERS.DOCCOUNT);

        Job job3 = Job.getInstance(conf, "Job 3");
        job3.setJarByClass(ProfileA.class);
        job3.setMapperClass(Mapper3.class);
        job3.setReducerClass(Reducer3.class);
        job3.setNumReduceTasks(1);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);

        if( !args[2].equals("1") )
        {
            job3.setPartitionerClass(UnigramPartitioner.class);
            job3.setNumReduceTasks(NUMBER_OF_REDUCERS_FOR_JOB_3);
        }
        else
        {
            job3.setNumReduceTasks(1);
        }

        //put counter value into conf object of the job where you need to access it
        //you can choose any name for the conf key really (i just used counter enum name here)
        job3.getConfiguration().setLong(CountersClass.N_COUNTERS.DOCCOUNT.name(), documentCount.getValue());
        FileInputFormat.addInputPath(job3, new Path(OUTPUT_PATH_FOR_JOB2));
        FileOutputFormat.setOutputPath(job3, new Path(args[1]));
        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }

}