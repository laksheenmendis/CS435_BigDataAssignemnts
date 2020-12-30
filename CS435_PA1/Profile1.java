import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.logging.Logger;

public class Profile1 {

    public static final int NUMBER_OF_REDUCERS = 1;
    public static final int FINAL_NO_OF_UNIGRAMS = 500;
    private static Logger LOGGER = Logger.getLogger(Profile1.class.getName());

    public static class UnigramMapper extends Mapper<Object, Text, Text, NullWritable> {

        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            //preprocessing of input
            if (!value.toString().isEmpty()) {
                StringTokenizer itr = new StringTokenizer(value.toString().split("<====>")[2]);

                while (itr.hasMoreTokens()) {

                    String out = itr.nextToken().replaceAll("[^A-Za-z0-9]", "").toLowerCase();

                    if( !out.isEmpty() )
                    {
                        word.set(out);
                        LOGGER.info( "M" + word.toString());

                        context.write( word, NullWritable.get() );
                    }
                }
            }
        }
    }

    public static class UnigramReducer extends Reducer< Text , NullWritable, Text, NullWritable> {

        private TreeMap<Text, Integer> reduceTMap =new TreeMap<Text, Integer>();

        @Override
        protected void reduce(Text key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {

            reduceTMap.put(new Text(key), 1);

            if (reduceTMap.size() > FINAL_NO_OF_UNIGRAMS) {
                reduceTMap.remove(reduceTMap.lastKey());
            }

            LOGGER.info( "Size of the map : " + reduceTMap.size() );

        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {

            LOGGER.info( "No of entries in the map :" + reduceTMap.size() );

            for (Map.Entry<Text, Integer> entry : reduceTMap.entrySet()) {
                context.write(entry.getKey(), NullWritable.get());
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "unigrams");
        job.setJarByClass(Profile1.class);
        job.setMapperClass(UnigramMapper.class);
        job.setReducerClass(UnigramReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);

        //set multiple reducers
        job.setNumReduceTasks(NUMBER_OF_REDUCERS);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
