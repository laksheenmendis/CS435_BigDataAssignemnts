import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.*;
import java.util.*;
import java.util.logging.Logger;

public class ProfileC {

    public static final int NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2 = 10;
    public static String SEPERATOR = "\t";
    private static Logger LOGGER = Logger.getLogger(ProfileC.class.getName());

    public static class Mapper1 extends Mapper<Object, Text, Text, Text> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String [] arr = value.toString().split(SEPERATOR);

            if( arr.length == 3 )
            {
                context.write( new Text(arr[0]), new Text( "A" + arr[1] + SEPERATOR + arr[2]));
            }
        }
    }

    public static class Mapper2 extends Mapper<Object, Text, Text, Text> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            if (!value.toString().isEmpty()) {
                String[] splitArr = value.toString().split("<====>");

                //check whether the body is there for a particular article
                if (splitArr.length == 3 && !splitArr[2].isEmpty()) {

                    context.write( new Text(splitArr[1]), new Text( "B" + splitArr[2] ) );

                }
            }
        }
    }

    public static class DocIDPartitioner extends Partitioner<Text, Text> {

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

    public static class Reducer1 extends Reducer< Text , Text, Text, NullWritable> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            String sentences = null;

            //{key, value} -> {(DocumentID\tunigram),TF-IDF}
            HashMap<String, Double> unigramTFIDFs = new HashMap<String, Double>();

            //{key, value} -> {Index,Sentence}
            HashMap<Integer, String> sentenceMap = new HashMap<Integer, String>();

            //{key, value} -> { Integer, Double}
            TreeMap<Integer, Double> sentenceTFIDFTreeMap = new TreeMap<Integer, Double>();

            //{key, value} -> {unigram, TFIDF}
            TreeMap<String, Double> unigramTreeMap = null;

            for (Text val : values) {

                if( val.toString().charAt(0) == 'A')
                {
                    String[] arr = val.toString().split(SEPERATOR);
                    unigramTFIDFs.put( key.toString() + SEPERATOR + arr[0].substring(1) , Double.parseDouble( arr[1] ) );
                }
                else if ( val.toString().charAt(0) == 'B')
                {
                    sentences = val.toString().substring(1);
                }
            }

            String documentID = key.toString();

            if( sentences != null )
            {
                //breaking into sentences
                String [] sentenceArr = sentences.split("\\. ");
                int arrLength = sentenceArr.length;

                int index = 0;
                for (String sentence : sentenceArr) {

                    if( !sentence.isEmpty() )
                    {
                        //we need to build the treeMap for each sentence
                        unigramTreeMap = new TreeMap<String, Double>();

//                            LOGGER.info("Document ID : " +documentID+"Sentence is :" + sentence);
                        sentenceMap.put( index, sentence.charAt(sentence.length() -1 ) == '.' ? sentence : sentence + "." );

                        StringTokenizer wordTknzr = new StringTokenizer(sentence);
                        while ( wordTknzr.hasMoreTokens() )
                        {
                            String unigram = wordTknzr.nextToken().replaceAll("[^A-Za-z0-9]", "").toLowerCase();

                            if ( !unigram.isEmpty() && !unigramTreeMap.keySet().contains(unigram) ) {

                                LOGGER.info("Unigram Val is :" + unigram + " " +index );

                                if( unigramTFIDFs.get(documentID+ SEPERATOR + unigram) != null )
                                {
                                    double tfidf = unigramTFIDFs.get(documentID+ SEPERATOR + unigram);
                                    unigramTreeMap.put( unigram , tfidf);
                                }
                            }
                        }

                        Map<String, Double> sortedMap = sortByValues(unigramTreeMap);

                        //TODO remove
                        StringBuilder sb1 = new StringBuilder( key + " Index: " + index );

                        //get the best 5 TFIDF values and calculate the sentence TF-IDF
                        double sentenceTFIDF = 0.00;
                        int count = 1;
                        for( Map.Entry entry : sortedMap.entrySet() )
                        {
                            sb1.append( entry.getKey() + " " + entry.getValue() + " ");
                            sentenceTFIDF += (Double)entry.getValue();
                            count += 1;
                            if(count == 6)
                            {
                                break;
                            }
                        }

                        LOGGER.info( "BEST 5 " + sb1);

//                            LOGGER.info("TREEMAP values : " + sortedMap.entrySet());
//                            LOGGER.info("Sentence : " + sentence);
                            LOGGER.info(key + " Index : " + index + " TFIDF : " + sentenceTFIDF);
                        //store the sentence TFIDF
                        sentenceTFIDFTreeMap.put( index, sentenceTFIDF );

                        index += 1;
                    }
                }

                Map<Integer, Double> sortedSentenceMap = sortByValues(sentenceTFIDFTreeMap);

                int count = 1;
                StringBuilder summaryBuilder = new StringBuilder("");
                List<Integer> indexList = new ArrayList<Integer>();

                for(Map.Entry entry: sortedSentenceMap.entrySet() )
                {
                    int sentenceIndex = (Integer) entry.getKey();
                    indexList.add(sentenceIndex);
                    count += 1;
                    if( count == 4  )
                    {
                        break;
                    }
                }

//                    LOGGER.info("Size of the index list is : " + indexList.size());
                //need to sort by index to preserve the original sequence of those sentences within the document
                Collections.sort(indexList);

                for( Integer indexVal : indexList )
                {
                    String sent = sentenceMap.get(indexVal);
                    summaryBuilder.append(sent);
                    summaryBuilder.append(" ");
                }

//                    LOGGER.info("Document : " +documentID + "Summary : " +summaryBuilder.toString());
                context.write( new Text(documentID + "<====>" + summaryBuilder.toString() ), NullWritable.get() );

            }

        }

        public static <K, V extends Comparable<V>> Map<K, V> sortByValues(final Map<K, V> map)
        {
            Comparator<K> valueComparator = new Comparator<K>() {
                public int compare(K k1, K k2) {
                    int compare =
                            map.get(k1).compareTo(map.get(k2));
                    if (compare == 0)
                        return 1;
                    else
                        return -compare;
                }
            };

            Map<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
            sortedByValues.putAll(map);
            return sortedByValues;
        }
    }


    public static void main(String[] args) throws Exception  {
        Configuration conf = new Configuration();
        Job job1 = Job.getInstance(conf, "Job 1");
        job1.setJarByClass(ProfileC.class);
        job1.setReducerClass(Reducer1.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(NullWritable.class);
        job1.setNumReduceTasks(1);

        MultipleInputs.addInputPath(job1, new Path(args[0]), TextInputFormat.class, Mapper1.class);
        MultipleInputs.addInputPath(job1, new Path(args[1]), TextInputFormat.class, Mapper2.class);

        FileOutputFormat.setOutputPath(job1, new Path(args[2]));

        //set multiple reducers
        if( !args[3].equals("1") )
        {
            job1.setPartitionerClass(DocIDPartitioner.class);
            job1.setNumReduceTasks(NUMBER_OF_REDUCERS_FOR_JOB_1_AND_2);
        }
        else
        {
            job1.setNumReduceTasks(1);
        }

        System.exit( job1.waitForCompletion(true) ? 0 : 1 );
    }

}
