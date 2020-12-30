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

public class Profile2 {

    public static final int NUMBER_OF_REDUCERS = 2;
    public static final int FINAL_NO_OF_UNIGRAMS = 500;
    public static final String OUTPUT_PATH_FOR_JOB1 = "./intermediate2";
    public static final String SEPERATOR = "\t";

    private static Logger LOGGER = Logger.getLogger(Profile2.class.getName());

    public static class Mapper1 extends Mapper<Object, Text, Text, Text> {

        //key -> {DocumentID~unigram}, value -> 1
        private Text word = new Text();
        private final static Text one = new Text("1");

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            //pre-processing of input
            if (!value.toString().isEmpty()) {

                String[] arr = value.toString().split("<====>");

                if (arr.length == 3) {

                    String documentID = arr[1];

                    StringTokenizer itr = new StringTokenizer(arr[2]);

                    while (itr.hasMoreTokens()) {

                        String unigram = itr.nextToken().replaceAll("[^A-Za-z0-9]", "").toLowerCase();

                        if(!unigram.isEmpty())
                        {
                            String outMapperKey = documentID + SEPERATOR + unigram;

                            word.set(outMapperKey);

                            context.write(word, one);
                        }
                    }
                }
            }
        }
    }

    public static class Reducer1 extends Reducer<Text, Text, Text, Text> {

        /*
        * The output {key, value} of the Reducer1 is {DocumentID,(unigram~frequency)}
        * */
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            int sum = 0;
            for (Text val : values) {
                sum += 1;
            }

            //process the key->DocumentID~unigram and separate DocumentID and Unigram
            String[] inputKeyArr = key.toString().split(SEPERATOR);

            if( inputKeyArr.length == 2 )
            {
//                LOGGER.info("R1***********Key:" +inputKeyArr[0] + "****Value:" +inputKeyArr[1] + SEPERATOR + String.valueOf(sum) );

                //output {key, value} -> {DocumentID, (unigram\tfrequency)}
                Text outputKey = new Text(inputKeyArr[0]);
                Text outputValue = new Text(inputKeyArr[1] + SEPERATOR + String.valueOf(sum));

                context.write(outputKey, outputValue);
            }
        }
    }

    public static class Mapper2 extends Mapper<LongWritable, Text, Text, NullWritable>{

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //here value would be in the form {DocumentID\tunigram~frequency}
            //hence, we need to do some preprocessing
            String [] arr = value.toString().split(SEPERATOR);

            if( arr.length == 3 )
            {
                LOGGER.info("*******M2****" + value);
                //here key would be in the form {DocumentID~unigram~frequency}
                context.write( new Text(value), NullWritable.get());
            }
        }
    }

    public static class Reducer2 extends Reducer<Text, NullWritable, Text, NullWritable> {

        private List<Text> list = null;
        private static Text documentID = null;
        private Map<String, Integer> countPerDoc;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            list = new ArrayList<Text>();
            countPerDoc = new HashMap<String, Integer>();
        }

        @Override
        protected void reduce(Text key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {

            String [] arr = key.toString().split(SEPERATOR);

            LOGGER.info("*********R2*****" + key.toString());

            if( countPerDoc.containsKey( arr[0] ) )
            {
                if( countPerDoc.get(arr[0]) < FINAL_NO_OF_UNIGRAMS )
                {
                    list.add( new Text(key) );
                    countPerDoc.put( arr[0], countPerDoc.get(arr[0]) + 1 );
                }
            }
            else
            {
                list.add( new Text(key) );
                countPerDoc.put( arr[0], 1 );
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {

            for( Text text: list )
            {
                context.write(text, NullWritable.get());
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

            Integer docId1 = Integer.parseInt(t1Items[0]);
            Integer docId2 = Integer.parseInt(t2Items[0]);

            String unigram1 = t1Items[1];
            String unigram2 = t2Items[1];

            Integer frequency1 = Integer.parseInt(t1Items[2]);
            Integer frequency2 = Integer.parseInt(t2Items[2]);

            Integer comparison = docId1.compareTo(docId2);

            if (comparison == 0) {
                comparison = -1 * frequency1.compareTo(frequency2);
                if (comparison == 0) {
                    comparison = unigram1.compareTo(unigram2);
                }
            }
            return comparison;
        }
    }

    public static class UnigramPartitioner extends Partitioner<Text, NullWritable> {

        @Override
        public int getPartition(Text key, NullWritable value, int numReduceTasks) {

//            if (numReduceTasks == NUMBER_OF_REDUCERS) {
                String [] arr = key.toString().split(SEPERATOR);

                //check document ID and decide the partition/reducer
                if( Integer.valueOf(arr[0]) % NUMBER_OF_REDUCERS == 1 )
                {
                    return 0;
                }
                else
                {
                    return 1;
                }
//            }
//            return 3;
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job1 = Job.getInstance(conf, "Job 1");
        job1.setJarByClass(Profile2.class);
        job1.setMapperClass(Mapper1.class);
//        job1.setPartitionerClass(UnigramPartitioner.class);
//        job1.setCombinerClass(Reducer1.class);
        job1.setReducerClass(Reducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        //set multiple reducers
        job1.setNumReduceTasks(NUMBER_OF_REDUCERS);
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(OUTPUT_PATH_FOR_JOB1));
        job1.waitForCompletion(true);

        Job job2 = Job.getInstance(conf, "Job 2");
        job2.setJarByClass(Profile2.class);
        job2.setMapperClass(Mapper2.class);
        job2.setPartitionerClass(UnigramPartitioner.class);
//        job2.setCombinerClass(Reducer2.class);
        job2.setReducerClass(Reducer2.class);

        //TODO check whether we should have a combiner, partitioner for Job 2

        job2.setSortComparatorClass(KeyComparator.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(NullWritable.class);
        job2.setNumReduceTasks(NUMBER_OF_REDUCERS);
        FileInputFormat.addInputPath(job2, new Path(OUTPUT_PATH_FOR_JOB1));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));
        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }

}
